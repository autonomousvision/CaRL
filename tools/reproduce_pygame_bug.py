import pygame

pygame.init()
surface = pygame.Surface((50000, 50000))

point1 = [49998, 32212]
point2 = [49999, 32212]
pygame.draw.line(surface, (255, 255, 255), point1, point2, 1)
print('Draw 1 worked')

point1 = [49998, 32212]
point2 = [49999, 32213]
pygame.draw.line(surface, (255, 255, 255), point1, point2, 1)
print('Draw 2 causes segfault.')
