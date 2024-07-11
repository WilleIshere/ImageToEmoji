from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from PIL import Image
import numpy as np
import random
import sys
import cv2
import os


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

__version__ = "0.1.0"

POPULATION_SIZE = 150
TARGET_IMAGE_PATH = "target_image.jpg"
STARTING_IMAGE_PATH = "starting.jpg"
GENERATION_DIRECTORY = "generations/"
EMOJIS_DIR = "emojis/"
MUTATIONS = 1
GENERATION_LIMIT = 15000
EMOJI_SIZE_MAX = 8  # Not used yet, the max size will be used for every emoji
EMOJI_SIZE_MIN = 8


def calculate_fitness(image, target_image):
    # Calculates returns a value closer to 1.0 based on how much the images match
    diff = np.sum(np.abs(image - target_image))
    scaled_diff = diff / (target_image.shape[0] * target_image.shape[1] * 255)
    fitness = 1 / (1 + scaled_diff)
    return fitness


def load_and_resize_emojis(emojis):
    # Resizes the emojis to a desired size
    emoji_images = {}
    for emoji_path in emojis:
        emoji_image = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
        emoji_image = cv2.resize(emoji_image, (EMOJI_SIZE_MAX, EMOJI_SIZE_MAX))
        emoji_images[emoji_path] = emoji_image
    return emoji_images


def mutate_image(image, emoji_images, image_size):
    # Randomly adds emojis to a set amount of images
    for mutation in range(random.randint(1, MUTATIONS)):
        emoji_path = random.choice(list(emoji_images.keys()))
        emoji_image = emoji_images[emoji_path]

        emoji_size = random.randint(EMOJI_SIZE_MIN, EMOJI_SIZE_MAX)
        x_offset = random.randint(0, image_size[1] - emoji_size)
        y_offset = random.randint(0, image_size[0] - emoji_size)

        overlay_color = emoji_image[:, :, :3]
        overlay_alpha = emoji_image[:, :, 3] / 255.0

        height, width = overlay_color.shape[:2]
        x_end = min(x_offset + width, image.shape[1])
        y_end = min(y_offset + height, image.shape[0])

        overlay_width = x_end - x_offset
        overlay_height = y_end - y_offset

        if overlay_width > 0 and overlay_height > 0:
            overlay_color = overlay_color[:overlay_height, :overlay_width]
            overlay_alpha = overlay_alpha[:overlay_height, :overlay_width]

            image_crop = image[y_offset:y_end, x_offset:x_end]

            if image_crop.shape[0] == overlay_height and image_crop.shape[1] == overlay_width:
                composite_color = (image_crop * (1 - overlay_alpha[..., np.newaxis]) +
                                   overlay_color * overlay_alpha[..., np.newaxis])

                image[y_offset:y_end, x_offset:x_end] = composite_color

    return image


def evolve(best_image, target_image, emoji_images):
    # Handles images and threading mutations to work faster on the CPU, and selects the best fitness image
    best = best_image
    image_list = [best.copy() for _ in range(POPULATION_SIZE)]
    image_size = best.shape

    while True:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(mutate_image, image.copy(), emoji_images, image_size) for image in image_list]
            new_image_list = [future.result() for future in as_completed(futures)]

        fitness_list = [calculate_fitness(image, target_image) for image in new_image_list]
        fitness = max(fitness_list)
        if fitness > calculate_fitness(best, target_image):
            break

    max_fitness_index = fitness_list.index(fitness)
    return fitness, new_image_list[max_fitness_index]


def render_text(screen, text, position, font, color=(255, 255, 255)):
    # Used for Generation and Fitness texts in the GUI
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def format_time_elapsed(start_time):
    elapsed_time = time() - start_time
    elapsed_time = abs(elapsed_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds = divmod(remainder, 1)[0]

    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def clear_line():
    terminal_width = os.get_terminal_size().columns
    print('\r' + ' ' * terminal_width + '\r', end='', flush=True)


def print_at_bottom(info_str):
    # Move cursor to bottom of terminal
    sys.stdout.write("\033[999;1H")  # Moves cursor to line 999, column 1 (bottom of terminal)
    sys.stdout.write(info_str)
    sys.stdout.flush()


def main():
    # The main loop pf the program
    import pygame as pg
    print("Starting...")
    pg.init()
    pg.font.init()

    print("Clearing previous generations...")
    if len(files := os.listdir(GENERATION_DIRECTORY)) > 0:
        for file in files:
            os.remove(GENERATION_DIRECTORY + file)

    print("Preparing images...")
    target_image = cv2.imread(TARGET_IMAGE_PATH)
    starting_image = cv2.imread(STARTING_IMAGE_PATH)
    emojis = [os.path.join(EMOJIS_DIR, x) for x in os.listdir(EMOJIS_DIR)]

    # Makes a white starting canvas
    target_size = target_image.shape
    starting_scale = starting_image.shape if starting_image is not None else (0, 0)
    if not target_size == starting_scale or not os.path.isfile(STARTING_IMAGE_PATH):
        if os.path.isfile(STARTING_IMAGE_PATH):
            os.remove(STARTING_IMAGE_PATH)
        img = Image.new("RGB", (target_size[1], target_size[0]))
        w, h = img.size
        for y in range(h):
            for x in range(w):
                img.putpixel((x, y), (255, 0, 0))

        img.save(STARTING_IMAGE_PATH)
        starting_image = cv2.imread(STARTING_IMAGE_PATH)
    starting_image = cv2.cvtColor(starting_image, cv2.COLOR_BGR2RGB)

    print("Scaling emojis...")
    # Resizes all the emojis
    emoji_images = load_and_resize_emojis(emojis)

    # Get a base image
    fitness, best_image = evolve(starting_image, target_image, emoji_images)

    # Initializes pygame to act as GUI
    print("Starting window...")
    screen = pg.display.set_mode((best_image.shape[1] * 2, best_image.shape[0] + 60))  # Extra space for text
    pg.display.set_caption("Emoji Evolution")

    font = pg.font.Font(None, 30)

    # Starting the main loop
    print("Starting evolution...")
    start_time = time()
    start_time_iterations = time()
    running = True
    generation = 0
    iterations = 0
    iterations_per_second = 0
    while running:
        generation += 1
        iterations += 1
        if generation >= GENERATION_LIMIT or fitness == 1.0:
            break

        best_image = cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB)
        fitness, best_image = evolve(best_image, target_image, emoji_images)

        current_time = time()
        elapsed_time = current_time - start_time_iterations
        if elapsed_time >= 0.25:
            iterations_per_second = iterations / elapsed_time
            iterations = 0
            start_time_iterations = current_time

        progress = generation / GENERATION_LIMIT
        time_elapsed = current_time - start_time
        if iterations_per_second > 0:
            estimated_total_time = time_elapsed / progress
            estimated_remaining_time = estimated_total_time - time_elapsed
        else:
            estimated_total_time = float('inf')
            estimated_remaining_time = float('inf')

        info_str = (
            "\r"
            f"| Generation: {(len(str(GENERATION_LIMIT)) - len(str(generation))) * ' '}{generation}/{GENERATION_LIMIT} "
            f"| Best Fitness: {fitness:.6f} "
            f"| Progress: {progress:.2%} "
            f"| Runtime: {format_time_elapsed(start_time)} "
            f"| ETA: {format_time_elapsed(time() + estimated_remaining_time)} "
            f"| Rate: {iterations_per_second:.2f} it/s |"
        )

        print_at_bottom(info_str)

        cv2.imwrite(f"{GENERATION_DIRECTORY}/{generation}.jpg", best_image)

        best_image = cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB)
        best_image_transposed = np.transpose(best_image, (1, 0, 2))
        pg_image = pg.surfarray.make_surface(best_image_transposed)

        target_image_ = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_image_transposed = np.transpose(target_image_, (1, 0, 2))
        pg_target_image = pg.surfarray.make_surface(target_image_transposed)

        screen.fill((0, 0, 0))
        screen.blit(pg_image, (0 - 1, 0))
        screen.blit(pg_target_image, (target_image.shape[1] + 1, 0))

        render_text(screen, f"Generation: {generation}", (10, best_image.shape[0] + 5), font)
        render_text(screen, f"Best Fitness: {fitness:.6f}", (10, best_image.shape[0] + 35), font)

        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False

        if not pg.display.get_active():
            print("Window not active, waiting for focus...")
            while not pg.display.get_active():
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                        break
                    elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                        running = False
                        break
                pg.time.wait(100)
    pg.quit()


if __name__ == '__main__':
    main()
