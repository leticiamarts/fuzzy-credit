# Simplified credit policy modeling using fuzzy logic
# For our simplified example, the score ranges from 0-1000, where 0 indicates a poor customer and 1000 a good customer.
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
import os

# Inputs
market_score = ctrl.Antecedent(np.arange(0, 1001, 1), "market_score")
internal_score = ctrl.Antecedent(np.arange(0, 1001, 1), "internal_score")
engagement = ctrl.Antecedent(np.arange(0, 5000, 1), "engagement")

# Output
customer_policy = ctrl.Consequent(np.arange(0, 1001, 1), "credit_policy")

# Fuzzification
market_score["RATING 1"] = fuzz.trimf(market_score.universe, [890, 900, 1000])
market_score["RATING 2"] = fuzz.trimf(market_score.universe, [750, 800, 890])
market_score["RATING 3"] = fuzz.trimf(market_score.universe, [570, 700, 750])
market_score["RATING 4"] = fuzz.trimf(market_score.universe, [390, 500, 570])
market_score["RATING 5"] = fuzz.trimf(market_score.universe, [240, 300, 390])
market_score["RATING 6"] = fuzz.trimf(market_score.universe, [0, 200, 240])
internal_score["RATING 1"] = fuzz.trimf(internal_score.universe, [890, 900, 1000])
internal_score["RATING 2"] = fuzz.trimf(internal_score.universe, [750, 800, 890])
internal_score["RATING 3"] = fuzz.trimf(internal_score.universe, [570, 700, 750])
internal_score["RATING 4"] = fuzz.trimf(internal_score.universe, [390, 500, 570])
internal_score["RATING 5"] = fuzz.trimf(internal_score.universe, [240, 300, 390])
internal_score["RATING 6"] = fuzz.trimf(internal_score.universe, [0, 200, 240])
engagement["high"] = fuzz.trimf(engagement.universe, [300, 1000, 5000])
engagement["medium"] = fuzz.trimf(engagement.universe, [100, 200, 300])
engagement["low"] = fuzz.trimf(engagement.universe, [0, 50, 100])

customer_policy["risk degree 1"] = fuzz.trimf(
    customer_policy.universe, [850, 950, 1000]
)
customer_policy["risk degree 2"] = fuzz.trimf(customer_policy.universe, [750, 850, 900])
customer_policy["risk degree 3"] = fuzz.trimf(customer_policy.universe, [650, 750, 800])
customer_policy["risk degree 4"] = fuzz.trimf(customer_policy.universe, [250, 500, 700])
customer_policy["risk degree 5"] = fuzz.trimf(customer_policy.universe, [0, 250, 300])

# Inference
rule1 = ctrl.Rule(
    engagement["low"] & (market_score["RATING 6"] | internal_score["RATING 6"])
    | (market_score["RATING 5"] | internal_score["RATING 5"])
    | (market_score["RATING 4"] | internal_score["RATING 4"])
    | (market_score["RATING 3"] | internal_score["RATING 3"]),
    customer_policy["risk degree 5"],
)
rule2 = ctrl.Rule(
    engagement["low"] & (market_score["RATING 1"] | internal_score["RATING 1"])
    | (market_score["RATING 2"] | internal_score["RATING 2"]),
    customer_policy["risk degree 4"],
)
rule3 = ctrl.Rule(
    engagement["medium"] & (market_score["RATING 6"] | internal_score["RATING 6"])
    | (market_score["RATING 5"] | internal_score["RATING 5"])
    | (market_score["RATING 4"] | internal_score["RATING 4"]),
    customer_policy["risk degree 5"],
)
rule4 = ctrl.Rule(
    engagement["medium"] & (market_score["RATING 1"] | internal_score["RATING 1"])
    | (market_score["RATING 2"] | internal_score["RATING 2"])
    | (market_score["RATING 3"] | internal_score["RATING 3"]),
    customer_policy["risk degree 4"],
)
rule5 = ctrl.Rule(
    engagement["high"] & (market_score["RATING 6"] | internal_score["RATING 6"]),
    customer_policy["risk degree 4"],
)
rule6 = ctrl.Rule(
    engagement["high"] & (market_score["RATING 5"] | internal_score["RATING 5"])
    | (market_score["RATING 4"] | internal_score["RATING 4"]),
    customer_policy["risk degree 3"],
)
rule7 = ctrl.Rule(
    engagement["high"] & (market_score["RATING 3"] | internal_score["RATING 3"]),
    customer_policy["risk degree 2"],
)
rule8 = ctrl.Rule(
    engagement["high"] & (market_score["RATING 1"] | internal_score["RATING 1"])
    | (market_score["RATING 2"] | internal_score["RATING 2"]),
    customer_policy["risk degree 1"],
)
policy_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
)
policy_simulator = ctrl.ControlSystemSimulation(policy_ctrl)

# Simulation
policy_simulator.input["market_score"] = 200
policy_simulator.input["internal_score"] = 230
policy_simulator.input["engagement"] = 90

# Defuzzification
policy_simulator.compute()
print(policy_simulator.output["credit_policy"])
output_value = policy_simulator.output["credit_policy"]
customer_policy.view(sim=policy_simulator)

base_filename = "credit_policy_plot"
output_folder = "output"
counter = 1

os.makedirs(output_folder, exist_ok=True)

while True:
    filename = os.path.join(output_folder, f"{base_filename}{counter}.png")
    if not os.path.exists(filename):
        break
    counter += 1

plt.savefig(filename)
