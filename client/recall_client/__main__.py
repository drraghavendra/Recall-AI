import argparse
import logging

from recall_client.config import ClientConfig


def _parse_args() -> argparse.Namespace:
    defaults = ClientConfig()
    parser = argparse.ArgumentParser(prog="python -m recall_client")
    parser.add_argument("--server", default=defaults.server_url, help="backend base URL")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="capture camera + mic and show personas (default)")
    run.add_argument("--camera", type=int, default=defaults.camera_index)
    run.add_argument("--window", type=float, default=defaults.window_s, help="sampling window seconds")
    run.add_argument("--frames", type=int, default=defaults.frames_per_window, help="frames per window (5-10)")
    run.add_argument("--persona-ttl", type=float, default=defaults.persona_ttl_s, help="persona cache TTL seconds")

    enroll = commands.add_parser("enroll-voice", help="record your voice and enroll it as 'self'")
    enroll.add_argument("--seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    if args.command == "enroll-voice":
        from recall_client.enroll import enroll_voice

        enroll_voice(args.server, args.seconds, ClientConfig().sample_rate)
        return

    from recall_client.app import run

    defaults = ClientConfig()
    run(
        ClientConfig(
            server_url=args.server,
            camera_index=getattr(args, "camera", defaults.camera_index),
            window_s=getattr(args, "window", defaults.window_s),
            frames_per_window=getattr(args, "frames", defaults.frames_per_window),
            persona_ttl_s=getattr(args, "persona_ttl", defaults.persona_ttl_s),
        )
    )


if __name__ == "__main__":
    main()
