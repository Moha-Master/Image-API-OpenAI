import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ModelScope/SiliconFlow Image API to OpenAI API Proxy"
    )
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/.config/image-api-openai/"),
        help="Working directory to read config.yaml from (default: ~/.config/image-api-openai/)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    args = parser.parse_args()

    work_dir = Path(args.dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    config_file = work_dir / "config.yaml"
    if not config_file.exists():
        example_file = Path(__file__).parent / "config.yaml.example"
        if example_file.exists():
            print(f"Error: config.yaml not found in {work_dir}")
            print(f"Please copy {example_file} to {config_file} and edit it.")
            sys.exit(1)
        print("Error: config.yaml.example not found in the package")
        sys.exit(1)

    import uvicorn

    uvicorn.run(
        "image_api_openai.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
