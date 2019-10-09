# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]



        "*** YOUR CODE HERE ***"
        my_score = 0
        if currentGameState.getFood()[newPos[0]][newPos[1]]: #newPos[0] = x  , newPos[1] = y
            my_score = my_score + 1 #Get score

        cur_pos = newPos

        while newFood:

            #Get the nearest food
            target_food = min(newFood, key=lambda x: manhattanDistance(x, cur_pos))
            min_dist = manhattanDistance(target_food, cur_pos)
            #Update the scores
            my_score = my_score + 1 / (min_dist + 1e-10)
            #remove food
            newFood.remove(target_food)
            #updata position( go to )
            cur_pos = target_food
        
        nearest_ghost = min(newGhostStates, key=lambda x: manhattanDistance(newPos, x.getPosition()))
        ghost_dist = manhattanDistance(newPos, nearest_ghost.getPosition())
        
        
        if ghost_dist < 3:
            return -100* (1/ghost_dist)

        return my_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        pacman_idx = 0
        other_idx = 1
        initial_depth = 0

        pacman_legal_actions = gameState.getLegalActions(pacman_idx)
        #Initialize to Null
        best_action  = None

        #Initialize to very small number.
        max_value = -10000

        for action in pacman_legal_actions:
            v = self.Min_Value(gameState.generateSuccessor(pacman_idx, action), other_idx, initial_depth)

            if v > max_value:
                best_action = action
                max_value = v
                

        return best_action

    def Max_Value (self, gameState, depth):
        
        Agentindex = 0
        otheridx = 1

        #If there is no legal actions
        if gameState.getLegalActions(Agentindex) == []:  
            return self.evaluationFunction(gameState)

        #If it is on the leaf 
        elif self.depth == depth:
            return self.evaluationFunction(gameState)

        #Choose the Max of the Minimizer
        else:
            return max([self.Min_Value(gameState.generateSuccessor(Agentindex, action), otheridx, depth) for action in gameState.getLegalActions(Agentindex)])

    def Min_Value (self, gameState, agentIndex, depth):

        #If there is no legal actions
        if gameState.getLegalActions(agentIndex) == []:        
            return self.evaluationFunction(gameState)

        elif agentIndex < gameState.getNumAgents() - 1 :
             return min([self.Min_Value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in gameState.getLegalActions(agentIndex)])
    
        #Choose The Min of the Maximizer
        else: 
            return min([self.Max_Value(gameState.generateSuccessor(agentIndex, action), depth + 1) for action in gameState.getLegalActions(agentIndex)])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        v = -10000
        #initialize as NULL
        best_action = None
        
        
        #initialize alpha as very small number and beta as very large number
        alpha = -10000
        beta = 10000



        # Just the same as MiniMax algorithm
        pacman_idx = 0
        other_idx = 1
        initial_depth = 0

        pacman_legal_actions = gameState.getLegalActions(pacman_idx)

        for action in pacman_legal_actions:
            v = self.Min_Value(gameState.generateSuccessor(pacman_idx, action), other_idx, initial_depth, alpha, beta)
            
            # if the value is larger than alpha
            if alpha < v:
                alpha = v
                best_action = action

        return best_action

    def Max_Value (self, gameState, depth, alpha, beta):

        pacman_idx = 0
        other_idx = 1
        v = -10000

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        elif gameState.getLegalActions(pacman_idx) == []:
            return self.evaluationFunction(gameState)


        for action in gameState.getLegalActions(0):
            v = max(v, self.Min_Value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta))

            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    def Min_Value (self, gameState, agentIndex, depth, alpha, beta):
        
        v = 10000
        # if depth == self.depth:
        #     return self.evaluationFunction(gameState)
        pacman_legal_actions = gameState.getLegalActions(agentIndex)

        if pacman_legal_actions == []:
            return self.evaluationFunction(gameState)

        else:
            for action in pacman_legal_actions:

                if agentIndex < gameState.getNumAgents() - 1:
                    v = min(v, self.Min_Value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                else:
                    v = min(v, self.Max_Value(gameState.generateSuccessor(agentIndex, action), depth + 1, alpha, beta))

                if v < alpha:
                    return v

                beta = min(beta, v)

            return v

 


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        pacman_idx = 0
        other_idx = 1
        initial_depth = 0

        pacman_legal_actions = gameState.getLegalActions(pacman_idx)
        #Initialize to Null
        best_action  = None

        #Initialize to very small number.
        max_value = -10000

        for action in pacman_legal_actions:
            v = self.Min_Value(gameState.generateSuccessor(pacman_idx, action), other_idx, initial_depth)

            if v > max_value:
                best_action = action
                max_value = v
                

        return best_action

    def Max_Value (self, gameState, depth):


        pacman_idx = 0
        other_idx = 1
        pacman_legal_actions = gameState.getLegalActions(pacman_idx)

        #If there is no legal actions
        if pacman_legal_actions == []:        
            return self.evaluationFunction(gameState)
            
        #If it is on the leaf 
        elif self.depth == depth:
            return self.evaluationFunction(gameState)
         #Choose the Max of the Minimizer
        else:
            return max([self.Min_Value(gameState.generateSuccessor(pacman_idx, action), other_idx, depth) for action in pacman_legal_actions])


    def Min_Value (self, gameState, agentIndex, depth):

        pacman_legal_actions = gameState.getLegalActions(agentIndex)

        if len(pacman_legal_actions) == 0:        
            return self.evaluationFunction(gameState)
        else:     
            a = 0   
            if agentIndex < gameState.getNumAgents() - 1:
                
                for action in pacman_legal_actions:
                    a = a + self.Min_Value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                return a / float(len(pacman_legal_actions))
            else:
                for action in pacman_legal_actions:
                    a = a + self.Max_Value(gameState.generateSuccessor(agentIndex, action), depth + 1)

                return a / float(len(pacman_legal_actions))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    

    score = 0


    cur_food = currentGameState.getPacmanPosition()
    food_list = (currentGameState.getFood()).asList()

    score = score - 5*len(food_list)

    #Add score if there is a food
    for food in food_list:
        new_food = min(food_list, key=lambda x: manhattanDistance(x, cur_food))
        score += 1.0/(manhattanDistance(cur_food, new_food))

        cur_food = new_food
        food_list.remove(new_food)
       

    cur_capsule = currentGameState.getPacmanPosition()
    capsule_list = currentGameState.getCapsules()

    score = score - 5*len(capsule_list)

    #Add score if there is a capsule
    for capsule in capsule_list:
        new_capsule = min(capsule_list, key=lambda x: manhattanDistance(x, cur_capsule))
        score += 1.0/(manhattanDistance(cur_capsule, new_capsule))

        cur_capsule = new_capsule
        capsule_list.remove(new_capsule)

    #Minus score if the ghosts are close
    if currentGameState.getNumAgents() > 1:
        ghost_dist = min( [ manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition()) 
                            for ghost in currentGameState.getGhostStates() ] )
        if ghost_dist < 2:
            return -10000
        else:    
            score = score - 1/ghost_dist

    score = score + 5*(currentGameState.getScore())

    return score

# Abbreviation
better = betterEvaluationFunction
