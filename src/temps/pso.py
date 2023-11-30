from random import uniform
import random

def cost_fcn(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2
    return total

class Particale:
    def __init__(self, x0):
        self.position_i = [] # position of the particle or agent
        self.velocity_i = [] # velocity of the particle or agent
        self.position_best_i = [] # determine the best position according to the particle's previous position
        self.error_best_i = -1 # best individual error
        self.err_i = -1 # individual error

        # init the velocity of par
        for i in range(0, num_dimensions):
            self.velocity_i.append(uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate each particle's position according to the objective function 
    def evaluate_fitness(self, cost_fcn):
        self.err_i = cost_fcn(self.position_i)
        # check to see if the current position is the individual best
        if self.err_i < self.error_best_i or self.error_best_i == -1:
            self.position_best_i = self.position_best_i
            self.error_best_i = self.err_i

    # update the particle's velocity
    def update_velocity(self, position_best_g):
        w = 0.5 # constant inertia weight 
        c1 = 1 # cognative constant 
        c2 = 1 # socail constant

        for i in range(0, num_dimensions):
            r1 = random.random() # random numbers
            r2 = random.random()
            # personal infleunce 
            vel_cognitive = c1 * r1 * (self.position_best_i[i] - self.position_i[i])
            # social influence 
            vel_social = c2 * r1 * (position_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w*self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle's position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]
            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]

def minimize(cost_fcn, x0, bounds, num_particles, maxiter, verbose):
        global num_dimensions

        num_dimensions = len(x0)
        err_best_g = -1      # best error for the group
        position_best_g = [] # best position for group 

        # create a population of particle swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particale(x0))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # loop through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate_fitness(cost_fcn)

                # determine if current particle is the best globally 
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    position_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)
                
                # loop through swarm and update velocities and position
                for j in range(0, num_particles):
                    swarm[j].update_velocity(position_best_g)
                    swarm[j].update_position(bounds)
            i += 1
        print('Final')
        print(position_best_g)
        print(err_best_g)

def test_everything():
    x0 = [5, 5]
    bounds = [(-10, 10), (-10, 10)]
    minimize(cost_fcn, x0, bounds, num_particles=15, maxiter=30, verbose=False)
    
test_everything()