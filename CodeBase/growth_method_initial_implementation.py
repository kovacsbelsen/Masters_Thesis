import pygame
import sys
import random
import math


# Constants
GRID_SIZE = 50
MIN_ROOM_WIDTH = 2
MIN_ROOM_HEIGHT = 2
ROOM_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
BACKGROUND_COLOR = (50, 50, 50)
GRID_COLOR = (100, 100, 100)
SCREEN_SIZE = (800, 800)
CELL_SIZE = SCREEN_SIZE[0] // GRID_SIZE
FPS = 10

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Floor Plan Generator")
clock = pygame.time.Clock()

# Initialize font
pygame.font.init()
font = pygame.font.Font(None, 24)

class RoomType:
    def __init__(self, name, min_size, max_size):
        self.name = name
        self.min_size = min_size
        self.max_size = max_size

room_types = [
    RoomType("Living Room", 60, 200),
    RoomType("Bedroom", 20, 80),
    RoomType("Kitchen", 20, 80),
    RoomType("Bathroom", 20, 80),
    RoomType("Entry Room", 20, 50),
    RoomType("Balcony", 25, 100),
]

class Apartment:
    def __init__(self, bounding_rect, rooms=None, desired_sizes=None):
        self.bounding_rect = bounding_rect
        self.rooms = rooms or []
        self.desired_sizes = desired_sizes or []    


    def calculate_bounding_rect(self):
        area_sqrt = int(math.sqrt(self.total_desired_area()))
        return pygame.Rect(0, 0, area_sqrt * CELL_SIZE, area_sqrt * CELL_SIZE)


    def total_desired_area(self):
        return sum(self.desired_sizes)

    def total_actual_area(self):
        return sum(room.width * room.height for room in self.rooms)

    def bounding_rect(self):
        min_x = min(room.x for room in self.rooms)
        max_x = max(room.x + room.width for room in self.rooms)
        min_y = min(room.y for room in self.rooms)
        max_y = max(room.y + room.height for room in self.rooms)
        return pygame.Rect(min_x * CELL_SIZE, min_y * CELL_SIZE, (max_x - min_x) * CELL_SIZE, (max_y - min_y) * CELL_SIZE)

    def draw(self, surface):
        bounding_rect = self.bounding_rect
        pygame.draw.rect(surface, (0, 0, 0), bounding_rect, 2)

    def __str__(self):
        return f"Apartment (Total Desired Area: {self.total_desired_area()}, Total Actual Area: {self.total_actual_area()})"

class Room:
    def __init__(self, id, x, y, width, height, color, room_type):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.room_type = room_type


    def __str__(self):
        return f"{self.room_type.name} (ID: {self.id}, X: {self.x}, Y: {self.y}, Width: {self.width}, Height: {self.height}, Color: {self.color})"


    def grow(self, dx, dy):
        self.x += min(dx, 0)
        self.y += min(dy, 0)
        self.width += abs(dx)
        self.height += abs(dy)

    def grow_left(self, grid, apartment_area):
        if self.x > 1 and apartment_area.collidepoint((self.x - 1) * CELL_SIZE, self.y * CELL_SIZE) and apartment_area.collidepoint((self.x - 1) * CELL_SIZE, (self.y + self.height - 1) * CELL_SIZE):
            for y in range(self.y, self.y + self.height):
                if grid.grid[self.x - 1][y] != -1:
                    return
            for y in range(self.y, self.y + self.height):
                grid.grid[self.x - 1][y] = self.id
            self.x -= 1
            self.width += 1
            
    def grow_right(self, grid, apartment_area):
        if self.x + self.width < grid.size - 1 and apartment_area.collidepoint((self.x + self.width) * CELL_SIZE, self.y * CELL_SIZE) and apartment_area.collidepoint((self.x + self.width) * CELL_SIZE, (self.y + self.height - 1) * CELL_SIZE):
            for y in range(self.y, self.y + self.height):
                if grid.grid[self.x + self.width][y] != -1:
                    return
            for y in range(self.y, self.y + self.height):
                grid.grid[self.x + self.width][y] = self.id
            self.width += 1

    def grow_up(self, grid, apartment_area):
        if self.y > 1 and apartment_area.collidepoint(self.x * CELL_SIZE, (self.y - 1) * CELL_SIZE) and apartment_area.collidepoint((self.x + self.width - 1) * CELL_SIZE, (self.y - 1) * CELL_SIZE):
            for x in range(self.x, self.x + self.width):
                if grid.grid[x][self.y - 1] != -1:
                    return
            for x in range(self.x, self.x + self.width):
                grid.grid[x][self.y - 1] = self.id
            self.y -= 1
            self.height += 1

    def grow_down(self, grid, apartment_area):
        if self.y + self.height < grid.size - 1 and apartment_area.collidepoint(self.x * CELL_SIZE, (self.y + self.height) * CELL_SIZE) and apartment_area.collidepoint((self.x + self.width - 1) * CELL_SIZE, (self.y + self.height) * CELL_SIZE):
            for x in range(self.x, self.x + self.width):
                if grid.grid[x][self.y + self.height] != -1:
                    return
            for x in range(self.x, self.x + self.width):
                grid.grid[x][self.y + self.height] = self.id
            self.height += 1



    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x * CELL_SIZE, self.y * CELL_SIZE, self.width * CELL_SIZE, self.height * CELL_SIZE), 0)
        
        # Draw the room type name centered in the room
        text_surface = font.render(self.room_type.name, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.center = ((self.x + self.width // 2) * CELL_SIZE, (self.y + self.height // 2) * CELL_SIZE)
        surface.blit(text_surface, text_rect)

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = [[-1 for _ in range(size)] for _ in range(size)]
        self.rooms = []

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def is_free(self, x, y, room):
        return self.in_bounds(x, y) and (self.grid[y][x] is None or self.grid[y][x] == room)

    def can_grow(self, room, dx, dy):
        for x in range(room.x + (room.width if dx > 0 else 0) + dx, room.x + room.width + dx):
            for y in range(room.y + (room.height if dy > 0 else 0) + dy, room.y + room.height + dy):
                if not self.is_free(x, y, room):
                    return False
        return True

    def add_room(self, room):
        for x in range(room.x, room.x + room.width):
            for y in range(room.y, room.y + room.height):
                if self.in_bounds(x, y):
                    self.grid[y][x] = room
        self.rooms.append(room)

    def grow_rooms(self, apartment_area, desired_sizes, apartment):
        growing = True
        max_iterations = 100  # Add a limit to the number of growing iterations
        current_iteration = 0  # Initialize a counter for the current iteration

        while growing and current_iteration < max_iterations:  # Update the loop condition
            growing = False
            current_iteration += 1  # Increment the counter
            for room, desired_size in zip(self.rooms, desired_sizes):
                if room.width * room.height < desired_size:
                    growing = True

                    # Choose a random direction for growth
                    direction = random.choice(["left", "right", "up", "down"])
                    if direction == "left":
                        room.grow_left(self, apartment_area)
                    elif direction == "right":
                        room.grow_right(self, apartment_area)
                    elif direction == "up":
                        room.grow_up(self, apartment_area)
                    elif direction == "down":
                        room.grow_down(self, apartment_area)

            # Draw the grid at every growth iteration
            screen.fill(BACKGROUND_COLOR)
            apartment.draw(screen)  # Draw the bounding rectangle of the apartment
            self.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)


    def draw(self, surface):
        for room in self.rooms:
            room.draw(surface)

        for x in range(self.size):
            for y in range(self.size):
                pygame.draw.line(surface, GRID_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, SCREEN_SIZE[1]), 1)
                pygame.draw.line(surface, GRID_COLOR, (0, y * CELL_SIZE), (SCREEN_SIZE[0], y * CELL_SIZE), 1)




def is_valid_room(room, apartment_area, grid):
    # Check if the room is inside the apartment area
    if not apartment_area.contains(pygame.Rect(room.x * CELL_SIZE, room.y * CELL_SIZE, room.width * CELL_SIZE, room.height * CELL_SIZE)):
        return False

    # Check if the room overlaps with other rooms
    for x in range(room.x, room.x + room.width):
        for y in range(room.y, room.y + room.height):
            if grid.grid[y][x] != -1:
                return False

    return True



def main():
    grid = Grid(GRID_SIZE)

    desired_sizes = [random.randint(room_type.min_size, room_type.max_size) for room_type in room_types]

    # Calculate the total desired area
    total_desired_area = sum(desired_sizes)
    
    # Calculate the size of the apartment area
    area_sqrt = int(math.sqrt(total_desired_area))
    apartment_area = pygame.Rect(0, 0, area_sqrt * CELL_SIZE, area_sqrt * CELL_SIZE)


    # Create and add rooms to the grid
    for i, (color, room_type) in enumerate(zip(ROOM_COLORS, room_types)):
        while True:
            x, y = random.randint(0, GRID_SIZE - MIN_ROOM_WIDTH), random.randint(0, GRID_SIZE - MIN_ROOM_HEIGHT)
            room = Room(i, x, y, MIN_ROOM_WIDTH, MIN_ROOM_HEIGHT, color, room_type)
            
            if is_valid_room(room, apartment_area, grid):
                break
                
        grid.add_room(room)



    # Create the apartment
    apartment = Apartment(apartment_area, rooms=grid.rooms)

    # Draw the apartment before growing the rooms
    screen.fill(BACKGROUND_COLOR)
    grid.draw(screen)
    apartment.draw(screen)
    pygame.display.flip()
    clock.tick(FPS)

    grid.grow_rooms(apartment_area, desired_sizes, apartment)

    print(apartment)  # Print apartment's information

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BACKGROUND_COLOR)
        grid.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()