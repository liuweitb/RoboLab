# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Pi0-family policy under ECM clarification.

ECM sees the scene image plus the (vague) task instruction and returns
``speak`` and ``intention``. The robot executes the intention, the user
confirms on stdin, then the policy finishes the task.

Needs ECM's package on the path: export PYTHONPATH="$ECM_PATH/src:$PYTHONPATH"
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]

parser = argparse.ArgumentParser(description="Evaluate a Pi0-family policy backend.")
parser.add_argument(
    "--policy",
    choices=PI0_VARIANTS,
    default="pi05",
    help=(
        "Which Pi0-family variant to evaluate (default: pi05). "
        "Selects per-variant defaults inside Pi0DroidJointposClient."
    ),
)
parser.add_argument(
    "--remote-host",
    "--remote_host",
    type=str,
    default="localhost",
    help="Remote host for policy server (default: localhost).",
)
parser.add_argument(
    "--remote-port", "--remote_port", type=int, default=8000, help="Remote port for policy server (default: 8000)."
)
parser.add_argument(
    "--remote-uri",
    "--remote_uri",
    type=str,
    default=None,
    help=(
        "Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
        "Overrides --remote-host and --remote-port when set."
    ),
)
parser.add_argument(
    "--open-loop-horizon",
    "--open_loop_horizon",
    type=int,
    default=None,
    help=(
        "Number of actions to execute from each predicted chunk before "
        "requesting a new one. If omitted, the client uses its per-variant "
        "default. Must match the model's action_horizon for best performance."
    ),
)
parser.add_argument(
    "--enable-verbose", "--enable_verbose", action="store_true", help="Verbose output (default: False)."
)
parser.add_argument("--enable-debug", "--enable_debug", action="store_true", help="Debug output (default: False).")
parser.add_argument(
    "--record-image-data",
    "--record_image_data",
    action="store_true",
    help="Enable proprio image data recording (default: False).",
)
parser.add_argument(
    "--randomize-background",
    "--randomize_background",
    action="store_true",
    help=(
        "Sample a random non-default background per task at registration time. "
        "Each registered env gets one fixed background; the chosen texture is "
        "recorded in the per-task env_cfg.json."
    ),
)
parser.add_argument(
    "--background-seed",
    "--background_seed",
    type=int,
    default=None,
    help="Seed for reproducible per-task background sampling. Used with --randomize-background.",
)

# ECM clarification
parser.add_argument(
    "--ecm-url",
    "--ecm_url",
    type=str,
    required=True,
    help="OpenAI-compatible base URL of the ECM server, e.g. http://localhost:8001/v1",
)
parser.add_argument("--ecm-model", "--ecm_model", type=str, required=True, help="Model name served by --ecm-url.")
parser.add_argument(
    "--indication-steps",
    "--indication_steps",
    type=int,
    default=100,
    help="Steps to execute ECM's intention before asking the user (default: 100).",
)
parser.add_argument(
    "--confirm-instruction",
    "--confirm_instruction",
    type=str,
    default=None,
    help="Instruction sent after the user confirms. Defaults to the task instruction.",
)
parser.add_argument(
    "--ecm-camera",
    "--ecm_camera",
    type=str,
    default="over_shoulder_left_camera",
    help="Camera fed to ECM (default: over_shoulder_left_camera, matches DROID).",
)
parser.add_argument(
    "--rephrase-url",
    "--rephrase_url",
    type=str,
    default=None,
    help="Base URL of the rephrase model (default: --ecm-url).",
)
parser.add_argument(
    "--rephrase-model",
    "--rephrase_model",
    type=str,
    default="Qwen/Qwen3.6-35B-A3B",
    help="Base model that rewrites the user's reply into an instruction.",
)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ecm.inference.inference import ECMClient, TranslatorClient  # noqa: E402

import robolab.constants  # noqa: E402
from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


class ECMClarifyingClient(Pi0DroidJointposClient):
    """pi0.5 driven by ECM: clarify -> indicate -> confirm -> execute.

    Assumes one env and one human at the terminal.
    """

    def __init__(
        self,
        ecm_url: str,
        ecm_model: str,
        indication_steps: int,
        confirm_instruction: str | None,
        ecm_camera: str,
        rephrase_url: str,
        rephrase_model: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ecm = ECMClient(model=ecm_model, base_url=ecm_url)
        self.indication_steps = indication_steps
        self.confirm_instruction = confirm_instruction
        self.ecm_camera = ecm_camera

        # Define the rephrasor from base model
        self.rephrasor = TranslatorClient(model=rephrase_model, base_url=rephrase_url)

        # Also set here, not just in begin_episode(), so the client is never in a
        # half-built state if something calls infer_batch() without an episode.
        self._reset_dialog_state()

    def _reset_dialog_state(self) -> None:
        self.phase = "clarify"
        self.instruction = None
        self.speak = ""
        self.steps = 0
        self.intention = ""

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._reset_dialog_state()

    def infer_batch(self, obs, instruction: str, *, env_ids: list[int]) -> dict[int, dict]:
        # The instruction argument is fixed for the episode; we swap in our own.
        if not env_ids:
            return {}
        if len(env_ids) > 1:
            raise RuntimeError(
                f"ECMClarifyingClient drives one terminal dialog and keeps a single "
                f"instruction for the whole batch, but got {len(env_ids)} envs. "
                f"Re-run with --num-envs 1."
            )
        if self.phase == "clarify":
            self.instruction = self._clarify(obs, instruction, env_ids[0])
        elif self.phase == "indicate" and self.steps >= self.indication_steps:
            self.instruction = self._confirm(obs, instruction, env_ids[0])
        self.steps += 1
        return super().infer_batch(obs, self.instruction, env_ids=env_ids)

    def _enter(self, phase: str, prompt: str) -> None:
        """Switch phase, dropping the action chunk queued under the old instruction.

        ``prompt`` is echoed because it, not the task instruction the runner prints
        at episode start, is what actually reaches the policy server.
        """
        self.phase = phase
        self.steps = 0
        super().reset()
        print(f"\033[96m[ECM][PHASE] {phase} -> policy prompt: {prompt!r}\033[0m", flush=True)

    def _clarify(self, obs, instruction: str, env_id: int) -> str:
        """Ask ECM what to say and do. On failure, just run the task instruction."""
        frame = unpack_image_obs(obs, env_id=env_id)[self.ecm_camera]
        try:
            reply = self.ecm.ask(frame, instruction)
        except Exception as error:
            # Loud and with the traceback: the quiet version of this path is
            # indistinguishable from ECM being skipped entirely, because the
            # fallback hands the policy the very instruction ECM exists to hide.
            print(
                f"\033[91m[ECM][ERROR] request to {self.ecm.model} failed "
                f"({type(error).__name__}: {error}).\n"
                f"[ECM][ERROR] FALLING BACK to the task instruction — the policy will "
                f"see {instruction!r} and no clarification will happen.\033[0m",
                flush=True,
            )
            traceback.print_exc()
            self._enter("execute", instruction)
            return instruction

        self.speak = reply.speak
        self.intention = reply.intention
        print(f"[ECM][INFO] speak    : {reply.speak}", flush=True)
        print(f"[ECM][INFO] intention: {reply.intention}", flush=True)
        self._enter("indicate", reply.intention)
        return reply.intention

    def _confirm(self, obs, instruction: str, env_id: int) -> str:
        """Take the user's reply. Yes -> finish the task; anything else -> translate it."""
        answer = input(f"[ECM] {self.speak}\n[user] ").strip()

        if answer.lower().startswith(("y", "correct", "right")):
            confirmed = self.confirm_instruction or instruction
            self._enter("execute", confirmed)
            return confirmed

        # A base model turns the reply into an unambiguous instruction for the policy.
        try:
            rephrased = self.rephrasor.translate(
                instruction, self.speak, self.intention, answer
            )
        except Exception as error:
            print(
                f"\033[91m[ECM][ERROR] translator {self.rephrasor.model} failed "
                f"({type(error).__name__}: {error}); running task instruction.\033[0m",
                flush=True,
            )
            traceback.print_exc()
            self._enter("execute", instruction)
            return instruction
        print(f"[ECM] rephrased: {rephrased}", flush=True)
        self._enter("execute", rephrased)
        return rephrased


def make_client(args: argparse.Namespace) -> ECMClarifyingClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant=args.policy,
    )
    return ECMClarifyingClient(
        ecm_url=args.ecm_url,
        ecm_model=args.ecm_model,
        indication_steps=args.indication_steps,
        confirm_instruction=args.confirm_instruction,
        ecm_camera=args.ecm_camera,
        rephrase_url=args.rephrase_url or args.ecm_url,
        rephrase_model=args.rephrase_model,
        **{k: v for k, v in kwargs.items() if v is not None},
    )


def main() -> None:
    # Label the run "ecm" so its output folder never collides with pi0 baselines.
    run_evaluation(args_cli, policy="ecm", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
