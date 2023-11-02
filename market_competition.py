import random
import matplotlib.pyplot as plt

# Markedsparametre

# Alpha: hvor meget forbrugere vil efterspørge, hvis prisen er nul
alpha = 10.5

# Beta: hældning på demand-kurve, hvor stor effekt prisen på varen har på efterspørgslen
beta = 1.75

# Theta: afgører hvilken indflydelse det har at prisen afviger fra markedsgennemsnittet
theta = 1.45833

def demand(price, market_price):
    return alpha - beta * price + theta * market_price

class Producer:
    def __init__(self, initial_price, initial_production):
        # Initielle værdier
        self.price = initial_price
        self.market_price_forecast = self.price + random.gauss(0, 5)
        self.quantity = initial_production
        self.excess_supply = 0

        # Normalfordelte konstanter (det står ikke i paperet hvordan de er sat)
        self.epsilon = random.gauss(0, 1)
        self.u = random.gauss(0, 1)
        self.eta = random.gauss(0, 1)

        # Koefficienter for agenten (pt. samme for alle agenter)
        # Disse er fra paperet, deltager 9, gruppe 4
        # beregnet via first-order heuristics

        # Koefficienter til estimering af markedspris
        self.w0 = 0.817 # vægtning af sidste obseverede markedspris
        self.w1 = 0.238 # vægtning af vores egen sidste forudsigelse

        # Koefficienter til beslutning af pris
        self.coeff_p = 0.832    # vægtning af sidste pris
        self.coeff_pe = 0.199   # vægtning af forventede markedspris
        self.coeff_S = -0.127  # straf for overskydende varer, der ikke bliver solgt

        # Koefficienter til beregning af produktion
        self.alpha = 11.812
        self.beta = 1.412
        self.theta = 1.058

    def set_price(self, market_price):
        # Forudsig fremtidig markedspris
        self.market_price_forecast = (self.w0 * market_price
                                      + self.w1 * self.market_price_forecast
                                      + self.epsilon)

        # Sæt vores pris denne periode
        self.price = (self.coeff_p * self.price
                      + self.coeff_pe * self.market_price_forecast
                      + self.coeff_S * self.excess_supply
                      + self.u)

    def set_production_level(self):
        # Forudsig efterspørgsel
        demand = (self.alpha
                  - self.beta * self.price
                  + self.theta * self.market_price_forecast
                  + self.eta)

        # Producer den mængde der efterspørges
        self.quantity = max(demand, 0)

    def observe_demand(self, demand):
        # Opdater overskud i varer (hvor meget blev ikke solgt)
        self.excess_supply = max(self.quantity - demand, 0)

# Et par hjælpefunktioner
def average(prices):
    return sum(prices)/len(prices)

def prices(agents):
    prices = []
    for agent in agents:
        prices.append(agent.price)
    return prices

def quantities(agents):
    quantities = []
    for agent in agents:
        quantities.append(agent.quantity)
    return quantities

# Parametre for simulationen
num_agents = 2000 # Antal agenter
iterations = 200 # Antal iterationer
initial_production_quantity = 20 # Alle agenter producerer lige meget

# Opret agenterne
agents = []
for i in range(num_agents):
    initial_price = random.gauss(15, 2)
    agent = Producer(initial_price, initial_production_quantity)
    agents.append(agent)

# Initiel markedspris
market_price = average(prices(agents))

# Gem markedspriser og produktions gennemsnit over tid (til plot)
market_price_list = [market_price]
production_averages = [initial_production_quantity]

# Kør simulationen
for t in range(iterations):
    # Lad agenterne sætte pris og produktionsniveau
    for agent in agents:
        agent.set_price(market_price)
        agent.set_production_level()

    # Beregn markedspris og gennemsnitlig produktion
    market_price = average(prices(agents))
    production_average = average(quantities(agents))
    
    # Lad agenter observere faktisk efterspørgsel
    for agent in agents:
        d = demand(agent.price, market_price)
        agent.observe_demand(d)

    # Gem markedspris og gns. produktion til plot
    market_price_list.append(market_price)
    production_averages.append(production_average)

# Print resultater
for t in range(iterations):
    print(t, market_price_list[t], production_averages[t])

# Plot markedspris
plt.title('markedspris vs. tid')
plt.xlabel('t')
plt.ylabel('markedspris')
plt.plot(range(iterations+1), market_price_list, 'm.')
plt.show()

# Plot produktion
plt.title('Gennemsnitlig produktion vs. tid')
plt.xlabel('t')
plt.ylabel('produktion')
plt.plot(range(iterations+1), production_averages, 'm.')
plt.show()
