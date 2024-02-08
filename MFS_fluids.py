from math import dist

class MFS_Particle():
    position = [0, 0]
    velocity = [0, 0]

    def speed():
        return dist(velocity[0], velocity[1])

class MFS_Solver():
    def initialize():
        pass

    def update():
        # USER NAVIER STOKES HERE TO ADJUST THE PARTICLES
        # SHOULD PROBABLY MAKE A CUSTOM CACHE FILE TYPE FOR POINTS TO READ / WRITE
        pass

    def reset():
        clear()
        initialize()
        pass

    def clear():
        pass

# MAKE A FUNCTION THAT DISTRIBUTES POINTS EVENLY (OR RANDOMLY) INSIDE OF AN OBJECT
# MAY NEED TO SET UP CUDA AND NUMPY FOR THIS

def point_distribute(pscale):
