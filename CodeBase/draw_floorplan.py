from Full_Project_Code_For_Thesis import ROOM_TYPES, build_slicing_tree_from_string, Floorplan, draw_floorplan, screen
import random
import pygame

# input the slicing tree to visualise the layout
slicingtree = 'H0.35(V0.50(R2bedR4kit)H0.61(V0.73(R1livR3bat)R5bed))'
a = build_slicing_tree_from_string(slicingtree, ROOM_TYPES)
room_types = [random.choice(ROOM_TYPES) for _ in range(5)]
floorplan = Floorplan(4, room_types, a)
draw_floorplan(floorplan.rooms, screen)
pygame.display.update()
print('*')