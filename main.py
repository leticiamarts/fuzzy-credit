# Modelagem de política de crédito simplificada usando lógica fuzzy
# Para o nosso exemplo simplificado o score varia de 0-1000, em que 0 indica um cliente ruim e 1000 um cliente bom.
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np

# Entradas
score_mercado = ctrl.Antecedent(np.arange(0, 1001, 1), "score_mercado")
score_interno = ctrl.Antecedent(np.arange(0, 1001, 1), "score_interno")
engajamento = ctrl.Antecedent(np.arange(0, 5000, 1), "engajamento")

# Saída
politica_cliente = ctrl.Consequent(np.arange(0, 1001, 1), "politica_de_risco")

# Fuzzificação
score_mercado["RATING 1"] = fuzz.trimf(score_mercado.universe, [890, 900, 1000])
score_mercado["RATING 2"] = fuzz.trimf(score_mercado.universe, [750, 800, 890])
score_mercado["RATING 3"] = fuzz.trimf(score_mercado.universe, [570, 700, 750])
score_mercado["RATING 4"] = fuzz.trimf(score_mercado.universe, [390, 500, 570])
score_mercado["RATING 5"] = fuzz.trimf(score_mercado.universe, [240, 300, 390])
score_mercado["RATING 6"] = fuzz.trimf(score_mercado.universe, [0, 200, 240])
score_interno["RATING 1"] = fuzz.trimf(score_interno.universe, [890, 900, 1000])
score_interno["RATING 2"] = fuzz.trimf(score_interno.universe, [750, 800, 890])
score_interno["RATING 3"] = fuzz.trimf(score_interno.universe, [570, 700, 750])
score_interno["RATING 4"] = fuzz.trimf(score_interno.universe, [390, 500, 570])
score_interno["RATING 5"] = fuzz.trimf(score_interno.universe, [240, 300, 390])
score_interno["RATING 6"] = fuzz.trimf(score_interno.universe, [0, 200, 240])
engajamento["alto"] = fuzz.trimf(engajamento.universe, [300, 1000, 5000])
engajamento["medio"] = fuzz.trimf(engajamento.universe, [100, 200, 300])
engajamento["baixo"] = fuzz.trimf(engajamento.universe, [0, 50, 100])

politica_cliente["grau de risco 1"] = fuzz.trimf(
    politica_cliente.universe, [850, 950, 1000]
)
politica_cliente["grau de risco 2"] = fuzz.trimf(
    politica_cliente.universe, [750, 850, 900]
)
politica_cliente["grau de risco 3"] = fuzz.trimf(
    politica_cliente.universe, [650, 750, 800]
)
politica_cliente["grau de risco 4"] = fuzz.trimf(
    politica_cliente.universe, [250, 500, 700]
)
politica_cliente["grau de risco 5"] = fuzz.trimf(
    politica_cliente.universe, [0, 250, 300]
)

# Inferência
regra1 = ctrl.Rule(
    engajamento["baixo"] & (score_mercado["RATING 6"] | score_interno["RATING 6"])
    | (score_mercado["RATING 5"] | score_interno["RATING 5"])
    | (score_mercado["RATING 4"] | score_interno["RATING 4"])
    | (score_mercado["RATING 3"] | score_interno["RATING 3"]),
    politica_cliente["grau de risco 5"],
)
regra2 = ctrl.Rule(
    engajamento["baixo"] & (score_mercado["RATING 1"] | score_interno["RATING 1"])
    | (score_mercado["RATING 2"] | score_interno["RATING 2"]),
    politica_cliente["grau de risco 4"],
)
regra3 = ctrl.Rule(
    engajamento["medio"] & (score_mercado["RATING 6"] | score_interno["RATING 6"])
    | (score_mercado["RATING 5"] | score_interno["RATING 5"])
    | (score_mercado["RATING 4"] | score_interno["RATING 4"]),
    politica_cliente["grau de risco 5"],
)
regra4 = ctrl.Rule(
    engajamento["medio"] & (score_mercado["RATING 1"] | score_interno["RATING 1"])
    | (score_mercado["RATING 2"] | score_interno["RATING 2"])
    | (score_mercado["RATING 3"] | score_interno["RATING 3"]),
    politica_cliente["grau de risco 4"],
)
regra5 = ctrl.Rule(
    engajamento["alto"] & (score_mercado["RATING 6"] | score_interno["RATING 6"]),
    politica_cliente["grau de risco 4"],
)
regra6 = ctrl.Rule(
    engajamento["alto"] & (score_mercado["RATING 5"] | score_interno["RATING 5"])
    | (score_mercado["RATING 4"] | score_interno["RATING 4"]),
    politica_cliente["grau de risco 3"],
)
regra7 = ctrl.Rule(
    engajamento["alto"] & (score_mercado["RATING 3"] | score_interno["RATING 3"]),
    politica_cliente["grau de risco 2"],
)
regra8 = ctrl.Rule(
    engajamento["alto"] & (score_mercado["RATING 1"] | score_interno["RATING 1"])
    | (score_mercado["RATING 2"] | score_interno["RATING 2"]),
    politica_cliente["grau de risco 1"],
)
politica_ctrl = ctrl.ControlSystem(
    [regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8]
)
politica_simulador = ctrl.ControlSystemSimulation(politica_ctrl)

# Simulação
politica_simulador.input["score_mercado"] = 200
politica_simulador.input["score_interno"] = 230
politica_simulador.input["engajamento"] = 90

# Deffuzificação
politica_simulador.compute()
print(politica_simulador.output["politica_de_risco"])
output_value = politica_simulador.output["politica_de_risco"]
politica_cliente.view(sim=politica_simulador)
plt.savefig("politica_de_risco_plot.png")
