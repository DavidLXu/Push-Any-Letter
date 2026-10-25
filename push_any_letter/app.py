"""Native macOS-compatible teleoperation window."""
import argparse
from pathlib import Path
import string
import pygame
import numpy as np
from .env import PushLetterEnv
from .render import draw_arena, ORANGE, TEAL
from .recording import Recorder

BG = (18, 29, 36)
PANEL = (27, 42, 50)
TEXT = (232, 240, 239)
MUTED = (145, 166, 174)


def main():
    parser = argparse.ArgumentParser(description="Push any letter into its target outline.")
    parser.add_argument("--letter", default="T", help="One character, e.g. T, a, Ω, 中")
    parser.add_argument("--font", help="TTF/OTF/TTC font containing the character")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "recordings"))
    parser.add_argument("--screenshot", help="Render the initial window to a PNG and exit")
    args = parser.parse_args()
    try:
        env = PushLetterEnv(letter=args.letter, font_path=args.font)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    pygame.init()
    screen = pygame.display.set_mode((1010, 768))
    pygame.display.set_caption("Push Any Letter | 2D Manipulation Lab")
    font = pygame.font.Font(None, 24)
    small = pygame.font.Font(None, 20)
    title = pygame.font.Font(None, 42)
    big = pygame.font.Font(None, 53)
    glyph_font = pygame.font.Font(env.glyph.font_path, 20)
    clock = pygame.time.Clock()
    seed = args.seed
    obs, info = env.reset(seed=seed)
    target = obs["agent_pos"].copy()
    recorder = Recorder(env, seed, source="human_mouse")
    recording = False
    paused = False
    ended = False
    lowercase = args.letter.islower()
    trail = []
    toast = "Hold left mouse on the table to push."
    buttons = []
    accumulator = 0.0

    def label(text, pos, color=TEXT, use_font=None):
        screen.blit((use_font or font).render(text, True, color), pos)

    def button(text, rect, key, active=False):
        r = pygame.Rect(rect)
        pygame.draw.rect(screen, (57, 93, 98) if active else PANEL, r, border_radius=8)
        rendered = (glyph_font if key.startswith("letter:") else small).render(text, True, TEXT)
        screen.blit(rendered, rendered.get_rect(center=r.center))
        buttons.append((r, key))

    def save_recording():
        nonlocal recorder, toast
        path = recorder.save(args.output)
        toast = f"Saved {len(recorder.actions)} steps to recordings/." if path else "No recorded steps yet."
        if path:
            print(f"Saved demonstration: {path}", flush=True)
        recorder = Recorder(env, seed, source="human_mouse")

    def reset(letter=None, same_seed=False):
        nonlocal seed, obs, info, target, recorder, ended, trail, toast, accumulator
        new_letter = letter or env.letter
        try:
            from .geometry import make_glyph
            make_glyph(new_letter, env.font_path)
        except ValueError as exc:
            toast = str(exc)
            return
        if recorder.actions:
            save_recording()
        if not same_seed:
            seed += 1
        obs, info = env.reset(seed=seed, options={"letter": new_letter})
        target = obs["agent_pos"].copy()
        recorder = Recorder(env, seed, source="human_mouse")
        ended, trail, accumulator = False, [], 0.0
        toast = "New scene. Hold left mouse on the table to push."

    running = True
    while running:
        dt = min(clock.tick(60) / 1000, 0.1)
        for event in pygame.event.get():
            key = None
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys = {pygame.K_ESCAPE: "quit", pygame.K_SPACE: "pause", pygame.K_r: "reset", pygame.K_n: "next", pygame.K_s: "save", pygame.K_c: "record", pygame.K_TAB: "case"}
                key = keys.get(event.key)
                if event.key == pygame.K_r and event.mod & pygame.KMOD_SHIFT:
                    key = "repeat"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                key = next((k for r, k in buttons if r.collidepoint(event.pos)), None)
            if key == "quit":
                running = False
            elif key == "pause":
                paused = not paused
            elif key in ("reset", "repeat"):
                reset(same_seed=key == "repeat")
            elif key == "case":
                lowercase = not lowercase
            elif key == "next":
                letters = string.ascii_lowercase if lowercase else string.ascii_uppercase
                reset(letters[(letters.find(env.letter)+1) % 26])
            elif key and key.startswith("letter:"):
                reset(key.split(":")[1])
            elif key == "record":
                if recording:
                    save_recording()
                # A fresh scene makes every recording replayable from its seed.
                else:
                    reset(same_seed=True)
                recording = not recording
                toast = "Recording ON. Stop or S to save." if recording else toast
            elif key == "save":
                save_recording()
                # Start a fresh seeded episode if recording continues.
                if recording:
                    reset(same_seed=True)
        mx, my = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0] and 28 <= mx < 668 and 98 <= my < 738:
            target = np.clip([mx-28, my-98], 28, 612).astype(np.float32)
        if not paused and not ended:
            accumulator += dt
            while accumulator >= 1 / env.control_hz and not ended:
                old_obs = obs
                obs, reward, terminated, truncated, info = env.step(target)
                if recording:
                    recorder.append(old_obs, target, obs, reward, terminated, truncated)
                trail.append(tuple(env.agent.position))
                trail = trail[-180:]
                accumulator -= 1 / env.control_hz
                ended = terminated or truncated
                if ended:
                    toast = "Target reached! Press R for a new scene." if terminated else "Episode finished. Press R to reset."
        else:
            accumulator = 0
        screen.fill(BG)
        buttons = []
        label("PUSH / ANY LETTER", (28, 23), use_font=title)
        label("2D MANIPULATION LAB     /     LOCAL PHYSICS", (30, 67), MUTED, small)
        pygame.draw.circle(screen, ORANGE if recording else TEAL, (780, 41), 4)
        label("REC" if recording else "CPU / LOCAL", (794, 33), MUTED, small)
        screen.blit(draw_arena(env, trail), (28, 98))
        label("GLYPH LIBRARY", (694, 101), MUTED, small)
        letters = string.ascii_lowercase if lowercase else string.ascii_uppercase
        for i, letter in enumerate(letters):
            button(letter, (694 + (i % 7)*41, 130+(i//7)*43, 35, 36), "letter:"+letter, letter == env.letter)
        button("a-z" if not lowercase else "A-Z", (694+5*41, 130+3*43, 76, 36), "case")
        label("TARGET COVERAGE", (694, 320), MUTED, small)
        label(f"{info['coverage']*100:05.1f}%", (692, 347), TEAL if info['is_success'] else TEXT, big)
        pygame.draw.rect(screen, PANEL, (694, 404, 280, 7), border_radius=3)
        if info['coverage'] > 0:
            pygame.draw.rect(screen, TEAL, (694, 404, max(3, int(280*info['coverage'])), 7), border_radius=3)
        label("GOAL 95%", (694, 424), MUTED, small)
        label(f"BEST {info['best_coverage']*100:.1f}%", (878, 424), MUTED, small)
        label(f"SEED {seed}    /    STEP {env.steps:03d}", (694, 464), MUTED, small)
        button("Resume [Space]" if paused else "Pause [Space]", (694, 501, 135, 39), "pause", paused)
        button("Reset [R]", (839, 501, 135, 39), "reset")
        button("Stop + save [C]" if recording else "Record [C]", (694, 550, 135, 39), "record", recording)
        button("Save [S]", (839, 550, 135, 39), "save")
        label("MOUSE   Hold + move to push", (694, 616), MUTED, small)
        label("N   Next letter     Tab   Letter case", (694, 641), MUTED, small)
        label("Shift+R   Repeat seed     Esc   Exit", (694, 666), MUTED, small)
        label("Orange: object    Teal: target", (694, 708), MUTED, small)
        label(toast[:110], (29, 746), MUTED, small)
        if paused or ended:
            badge = "PAUSED" if paused else ("TARGET REACHED" if info["is_success"] else "EPISODE COMPLETE")
            pygame.draw.rect(screen, BG, (233, 112, 240, 34), border_radius=8)
            label(badge, (249, 121), TEAL if ended else TEXT, small)
        pygame.display.flip()
        if args.screenshot:
            pygame.image.save(screen, args.screenshot)
            running = False
    if recorder.actions:
        save_recording()
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
