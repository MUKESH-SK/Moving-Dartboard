import pygame

def boardmovement(x_initial,y_initial,target_x,target_y):
    # Initialize Pygame
    pygame.init()

    # Set the window size
    size = (700, 500)
    screen = pygame.display.set_mode(size)

    # Set the circle's starting position
    # x_initial = 50
    # y_initial = 50
    #
    # # Set the target coordinate
    # target_x = 700
    # target_y = 500

    # Set the speed of the 's movement
    speed = 0.5

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move the circle towards the target coordinate
        if y_initial < target_y:
            y_initial += speed
        if y_initial > target_y:
            y_initial -= speed
        if x_initial < target_x:
            x_initial += speed
        if x_initial > target_x:
            x_initial -= speed

        # Clear the screen
        screen.fill((255, 255, 255))

        # Draw the circle
        pygame.draw.circle(screen, (0, 0, 255), (x_initial, y_initial), 20)

        # Update the display
        pygame.display.flip()

    # Exit Pygame
    pygame.quit()

boardmovement(50,50,700,500)
boardmovement(700,500,10,400)