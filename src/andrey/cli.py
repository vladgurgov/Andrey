"""CLI entry point for the Andrey game testing assistant."""

from typing import Optional

import click

from andrey.config import load_config
from andrey.device import DeviceManager
from andrey.logger import setup_logging
from andrey.vision import VisionClient


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config: Optional[str], verbose: bool):
    """Andrey - AI-powered Android game testing assistant."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose
    # Defer setup_logging — play command adds log_dir, other commands call it here
    if ctx.invoked_subcommand != "play":
        setup_logging(verbose=verbose)


@main.command()
@click.option("--profile", "-p", default=None, help="Game profile name (e.g., 'spades')")
@click.option("--device", "-d", default=None, help="Device serial number")
@click.option("--delay", type=float, default=None, help="Delay between steps (seconds)")
@click.option("--steps", "-n", type=int, default=None, help="Maximum number of steps (actions)")
@click.option("--context", type=str, default=None, help="Extra context for the LLM")
@click.option("--max-images", type=int, default=None, help="Max screenshots in conversation context")
@click.option("--omniparser-path", type=str, default=None, help="Path to OmniParser repo")
@click.option("--no-omniparser", is_flag=True, help="Disable OmniParser element detection")
@click.option("--save-annotated", is_flag=True, help="Save OmniParser annotated screenshots for debugging")
@click.pass_context
def play(ctx, profile, device, delay, steps, context, max_images,
         omniparser_path, no_omniparser, save_annotated):
    """Start playing a game automatically."""
    from andrey.agent import AgentLoop

    config = ctx.obj["config"]

    if profile:
        config.game_profile = profile
    if device:
        config.device.serial = device
    if delay is not None:
        config.loop.delay_seconds = delay
    if steps is not None:
        config.loop.max_steps = steps
    if max_images is not None:
        config.conversation.max_images = max_images
    if omniparser_path:
        config.omniparser.omniparser_path = omniparser_path
    if no_omniparser:
        config.omniparser.enabled = False
    if save_annotated:
        config.save_annotated = True

    # Setup logging with file output to screenshot dir
    setup_logging(verbose=ctx.obj["verbose"], log_dir=config.screenshot_dir)

    if not config.anthropic.api_key:
        raise click.ClickException(
            "No API key found. Set ANTHROPIC_API_KEY env var or add it to config.yaml."
        )

    agent = AgentLoop(config)
    agent.run(extra_context=context)


@main.command()
@click.option("--device", "-d", default=None, help="Device serial number")
@click.option("--save", "-s", default=None, help="Save screenshot to this path")
@click.pass_context
def screenshot(ctx, device, save):
    """Take a screenshot and describe what's on screen."""
    config = ctx.obj["config"]
    if device:
        config.device.serial = device

    dm = DeviceManager(
        serial=config.device.serial,
        adb_host=config.device.adb_host,
        adb_port=config.device.adb_port,
    )
    dm.connect()

    img = dm.screenshot()

    if save:
        img.save(save)
        click.echo(f"Screenshot saved to {save}")

    if not config.anthropic.api_key:
        click.echo("No API key set. Screenshot saved but cannot describe without it.")
        return

    vision = VisionClient(
        api_key=config.anthropic.api_key,
        model=config.anthropic.model,
    )
    description = vision.describe_screenshot(img)
    click.echo(f"\n{description}")


@main.command()
@click.argument("x", type=int)
@click.argument("y", type=int)
@click.option("--device", "-d", default=None, help="Device serial number")
@click.pass_context
def tap(ctx, x, y, device):
    """Manually tap at the given coordinates."""
    config = ctx.obj["config"]
    if device:
        config.device.serial = device

    dm = DeviceManager(
        serial=config.device.serial,
        adb_host=config.device.adb_host,
        adb_port=config.device.adb_port,
    )
    dm.connect()
    dm._device.click(x, y)
    click.echo(f"Tapped at ({x}, {y})")


@main.command()
@click.pass_context
def devices(ctx):
    """List connected ADB devices."""
    config = ctx.obj["config"]
    serials = DeviceManager.list_devices(
        adb_host=config.device.adb_host,
        adb_port=config.device.adb_port,
    )
    if not serials:
        click.echo("No devices connected.")
    else:
        click.echo(f"Connected devices ({len(serials)}):")
        for s in serials:
            click.echo(f"  - {s}")
