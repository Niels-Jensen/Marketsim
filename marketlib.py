import urllib
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm

github_url = "https://raw.githubusercontent.com/DatalogiForAlle/market_competition/master"

def get_parameters(mp, agent_type = "empirical", num_agents = 0, std_scale = 0.1, pe_means = {}, p_means = {}, q_means = {}, use_actual_agents = True):
   
    # -----------------------------------------------------------------------------------------    
    # Her defineres agenter ud fra de estimerede ligninger i artiklen.
    # Man kan vælge om alle grupperne eller blot et subset anvendes. 
    # Hvis only_actual_agents == True, vil man anvende præcis de estimater, der angives i artiklen.
    # Ellers vil man sample et antal svarende til num_agents fra artiklens estimater.  
    if (agent_type == "empirical"):

        # Hent parametre som .csv-filer
        urllib.request.urlretrieve(github_url + "/p_params.csv", "p_params.csv")
        urllib.request.urlretrieve(github_url + "/pe_params.csv", "pe_params.csv")
        urllib.request.urlretrieve(github_url + "/q_params.csv", "q_params.csv")

        # Indlæs .csv filer
        pe_params = pd.read_csv('pe_params.csv')
        p_params = pd.read_csv('p_params.csv')
        q_params = pd.read_csv('q_params.csv')

        # Udvælg alle parametre fra det første eksperiment, alle grupper
        group_id = list(set(pe_params["Group"]))
        # Alternativt: Udvælg alle parametre fra det første eksperiment, gruppe i,j..
        # group_id = [1]

        # Subset individers parametre, svarende til grupperne specificeret ovenfor
        pe_params = pe_params[pe_params["Group"].isin(group_id)]
        p_params = p_params[p_params["Group"].isin(group_id)]
        q_params = q_params[q_params["Group"].isin(group_id)]
        pe_params.reset_index()
        p_params.reset_index()
        q_params.reset_index()

        # Anvend kun de estimerede faktiske agenter i artiklen - kan bruges til genskabelse af artiklens grafer. 
        # Hvis falsk anvendes syntetiske samplede værdier i stedet.
        only_actual_agents = use_actual_agents

        if (only_actual_agents):
            return pe_params, p_params, q_params, pe_params.shape[0]

        # Tag gennemsnit af parameterværdierne til fra artiklen til sampling fordeling 
        pe_means = dict(pe_params[["c", "alpha_1", "alpha_2", "alpha_3"]].mean())
        p_means = dict(p_params[["c", "beta_1", "beta_2", "beta_3", "beta_4", "diff_pi", "diff_p"]].mean())
        q_means = dict(q_params[["c", "gamma_1", "gamma_2", "gamma_3", "gamma_4"]].mean())

    # -----------------------------------------------------------------------------------------    
    # Her defineres agenter som reagerer rationelt. 
    # Dvs. de responderer optimalt givet modellens parametre på den observerede gennemsnitspris    
    elif (agent_type == "profit_seeking"):


        alpha_p = mp['alpha']/(2*mp['beta'])
        theta_p = mp['theta']/(2*mp['beta'])
        c_p = alpha_p + mp['c']/2

        # Note: funktionen nedenfor viser en producents best-response pris givet gennemsnitsprisen p_bar
        # (Denne funktion ligger implicit i p_means) 
        # p_br = lambda p_bar: alpha_p + mp['c']/2 + theta_p*p_bar

        if len(pe_means) == 0:  
            pe_means = {"c": 0.0,
                        "alpha_1": 0.5,
                        "alpha_2": 0.5,
                        "alpha_3": 0.0}

        if len(p_means) == 0:
            p_means = {"c": c_p, 
                        "beta_1": 0.0, 
                        "beta_2": theta_p, 
                        "beta_3": 0.0, 
                        "beta_4": 0.0, 
                        "diff_pi": 0.0, 
                        "diff_p": 0.0}

        if len(q_means) == 0:            
            q_means = {"c": mp['alpha'], 
                        "gamma_1": 0.0, 
                        "gamma_2": -mp['beta'], 
                        "gamma_3": mp['theta'], 
                        "gamma_4": 0.0}

    elif (agent_type == "herding"):

        if len(pe_means) == 0:  
            pe_means = {"c": 0.0,
                        "alpha_1": 0.5,
                        "alpha_2": 0.5,
                        "alpha_3": 0.0}

        if len(p_means) == 0:
            p_means = {"c": 0, 
                        "beta_1": 0.5, 
                        "beta_2": 0.5, 
                        "beta_3": 0.0, 
                        "beta_4": 0.0, 
                        "diff_pi": 0.0, 
                        "diff_p": 0.0}

        if len(q_means) == 0:    
            q_means = {"c": mp['alpha'], 
                        "gamma_1": 0.0, 
                        "gamma_2": -mp['beta'], 
                        "gamma_3": mp['theta'], 
                        "gamma_4": 0.0}

    else:
       raise ValueError('get_parameters: Ugyldigt agent_type argument')


    # Standard afvigelser af pris-forventningsparametre er proportionale
    pe_stds = dict(pe_means)        
    for j in pe_stds:
        pe_stds[j] *= std_scale

    p_stds = dict(p_means)
    for j in p_stds:
        p_stds[j] *= std_scale 

    q_stds = dict(q_means)
    for j in q_stds:
        q_stds[j] *= std_scale 


    # Træk parameterværdier for hver agent fra de definerede fordelinger. 
    # Bemærk: trunkering af fordelingerne for at undgå ekstreme størrelser

    # Antal standardafvigelser før trunkering på hver side af gennemsnit
    a = -1
    b = 1

    # -------------------------------
    # Parametre i pris-forventning
    # -------------------------------

    # Trunkeret fordeling
    rnd = truncnorm.rvs(a, b, size = (num_agents, len(pe_means)))

    # Reshape data strukturer og anvend gennemsnit + tilfældige stræk til at skabe parametre
    means = np.tile(list(pe_means.values()), (num_agents, 1))
    stds = np.tile(list(pe_stds.values()), (num_agents, 1))
    df = pd.DataFrame(means + rnd*stds)

    # Omdøb kolonner
    colnames = dict(zip(list(range(len(pe_means.keys()))), pe_means.keys()))
    pe_params = df.rename(columns = colnames)


    # -------------------------------
    # Parametre i prissætning
    # -------------------------------

    # Trunkeret fordeling
    rnd = truncnorm.rvs(a, b, size = (num_agents, len(p_means)))

    # Reshape data strukturer og anvend gennemsnit + tilfældige stræk til at skabe parametre
    means = np.tile(list(p_means.values()), (num_agents, 1))
    stds = np.tile(list(p_stds.values()), (num_agents, 1))
    df = pd.DataFrame(means + rnd*stds)

    # Omdøb kolonner
    colnames = dict(zip(list(range(len(p_means.keys()))), p_means.keys()))
    p_params = df.rename(columns = colnames)

    # Tilføj type af agent, agent_type
    p_params['agent_type'] = [agent_type for i in range(p_params.shape[0])]


    # -----------------------------------
    # Parametre i produktionsbeslutning
    # -----------------------------------

    # Trunkeret fordeling
    rnd = truncnorm.rvs(a, b, size = (num_agents, len(q_means)))

    # Reshape datastrukturer og anvend gennemsnit + tilfældige stræk til at skabe parametre
    means = np.tile(list(q_means.values()), (num_agents, 1))
    stds = np.tile(list(q_stds.values()), (num_agents, 1))
    df = pd.DataFrame(means + rnd*stds)

    # Omdøb kolonner
    colnames = dict(zip(list(range(len(q_means.keys()))), q_means.keys()))
    q_params = df.rename(columns = colnames)

    num_agents = pe_params.shape[0]    

    return pe_params, p_params, q_params, num_agents

# Agent class
class Producer:
    def __init__(self, initial_price, initial_production, endowment, c, pe, p, q, epsilon = 0, u = 0, eta = 0):
        # Initiale værdier
        self.price = initial_price
        self.price_t1 = initial_price # pris ved t-1
        self.price_t2 = initial_price # pris ved t-2
        
        self.market_price_forecast = self.price
        self.quantity = initial_production
        self.excess_supply = 0
        self.sold_goods = 0
        
        self.profit = 0
        self.profit_t1 = 0 # profit t-1
        self.profit_t2 = 0 # profit t-2
        self.price_adjustment = 0 # Pi i paperet

        # Vi sætter konstanterne epsilon, u, eta til 0 (paperet fortæller ikke hvordan de er sat)
        self.epsilon = epsilon
        self.u = u
        self.eta = eta

        # Koefficienter for agenten baseret på estimater
        self.pe = pe
        self.p = p
        self.q = q

        # Hvilken type har producent-agenten (eks. 'best-response', 'herding', ...)
        self.agent_type = p.agent_type
        
        # Marginal produktionsomkostning
        self.mc = c
        
        # Balance is set to initial endowment
        self.balance = endowment
        self.bankrupt = False

    def set_price(self, market_price_t1, market_price_t2):
        if self.bankrupt:
            self.price = np.nan
            return
        
        # Forudsig indeværende periodes gennemsnitlige pris
        self.market_price_forecast = (self.pe.c
                                      + self.pe.alpha_1 * market_price_t1
                                      + self.pe.alpha_2 * self.market_price_forecast
                                      + self.pe.alpha_3 * market_price_t2
                                      + random.gauss(0, self.epsilon))

        # Opdater historiske priser
        self.price_t2 = self.price_t1
        self.price_t1 = self.price
        
        # Sæt vores pris denne periode
        self.price = (self.p.c 
                      + self.p.beta_1 * self.price
                      + self.p.beta_2 * self.market_price_forecast
                      + self.p.beta_3 * self.price_adjustment
                      + self.p.beta_4 * self.excess_supply
                      + random.gauss(0, self.u))

    def set_production_level(self):
        if self.bankrupt:
            self.quantity = np.nan
            return
        
        # Estimer efterspørgsel
        estimated_demand = (self.q.c
                            + self.q.gamma_1 * self.quantity
                            + self.q.gamma_2 * self.price
                            + self.q.gamma_3 * self.market_price_forecast
                            + self.q.gamma_4 * self.excess_supply
                            + random.gauss(0, self.eta))

        # Producer den mængde vi forventer der efterspørges
        self.quantity = max(estimated_demand, 0)

    def observe_demand(self, demand):
        if self.bankrupt:
            self.excess_supply = np.nan
            self.sold_goods = np.nan
            return
        
        # Opdater overskud i varer (hvor meget blev ikke solgt)
        self.sold_goods = min(self.quantity, demand)
        
        # Her er excess supply forskel mellem produktion og solgt mængde. Kan kun være positiv. 
        # self.excess_supply = self.quantity - self.sold_goods

        # Her er excess supply defineret som forskel mellem produktion og efterspørgsel (ikke den faktisk solgte mængde). 
        # Kan derfor både være positiv og negativ. 
        self.excess_supply = self.quantity - demand

    
    def calculate_profit(self):
        if self.bankrupt:
            self.profit = np.nan
            return
        
        # Opdater historisk profit
        self.profit_t2 = self.profit_t1
        self.profit_t1 = self.profit

        # Beregn profit i denne periode
        self.profit = self.price*self.sold_goods - self.mc*self.quantity
        
        # Opdater balance
        self.balance += self.profit
        
        # Er vi gået konkurs?
        self.bankrupt = self.balance < 0

    def update_price_adjustment(self):
        # Korriger pricer i den retning som giver højere profit
        if self.bankrupt:
            self.price_adjustment = np.nan
            return
        
        # Ændring i pris i forrige periode
        price_difference = self.price_t1 - self.price_t2
        
        if self.profit_t1 < self.profit_t2:
            # Hvis profitten er gået ned, justerer vi prisen længere ned
            self.price_adjustment = - price_difference
        else:
            # Hvis profiten er gået op, justerer vi prisen længere op
            self.price_adjustment = price_difference

    def p_br(self, mp, p_bar):
        # Best response pris - rationel adfærd    

        alpha_p = mp['alpha']/(2*mp['beta'])
        theta_p = mp['theta']/(2*mp['beta'])
        c_p = alpha_p + mp['c']/2

        return alpha_p + mp['c']/2 + theta_p*p_bar
         