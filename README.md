# Credit Policy Modeling with Fuzzy Logic

This Python script demonstrates a simplified credit policy modeling using fuzzy logic. The script uses the scikit-fuzzy library to define fuzzy variables, rules, and perform inference to determine a credit policy based on input variables.

## Installation

Before running the script, make sure to install the required libraries. It's recommended to create a virtual environment to manage these libraries separately. Here's how to set up the environment:

1. **Create a Virtual Environment (Optional)**

   You can create a virtual environment using Python's built-in `venv` module. If you're not familiar with `venv`, you can refer to the [official Python documentation](https://docs.python.org/3/library/venv.html) for more information on how to create and manage virtual environments.

2. **Install Required Libraries**

```
pip install numpy scikit-fuzzy matplotlib
```

## Usage

To vary the output result, you can adjust the input variables: `market_score`, `internal_score`, and `engagement`. These variables represent external market score, internal score, and customer engagement, respectively. Modify their values as needed to observe different outcomes of the credit policy.

```python
# Example: Vary the input variables
policy_simulator.input["market_score"] = 200
policy_simulator.input["internal_score"] = 230
policy_simulator.input["engagement"] = 90
```

## Running the Script

Run the Python script credit_policy.py to perform the credit policy modeling and view the output. The script will also save the resulting plot in the output folder with a unique filename to avoid overwriting previous results.

```
python credit_policy.py
```

## References

For more details on the application of fuzzy logic in financial credit analysis, you can refer to the following article:

- [Applying Fuzzy Logic in Financial Credit Analysis](https://medium.com/datarisk-io/aplicando-l%C3%B3gica-fuzzy-na-an%C3%A1lise-financeira-de-cr%C3%A9dito-6b728cd46abc)

Please visit the provided link for additional insights and information related to this project.
