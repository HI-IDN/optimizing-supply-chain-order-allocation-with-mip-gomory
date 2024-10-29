# Optimizing Supply Chain Order Allocation with Mixed Integer Programming

## Context:

In this project, you'll leverage Mixed Integer Programming (MIP) to solve an order allocation
problem. The goal is to assign a set of customer orders to production plants and transportation
routes, minimizing the total cost while adhering to the restrictions of the supply chain structure.

## Problem Description:

For each customer order, determine the optimal plant to process it and the most efficient port for
shipment.

### Requirements:

- **Plant Selection**: Each plant can only process specific types of products.
- **Port Assignment**: Plants are restricted to shipping only through certain connected ports.
- **Vendor Managed Inventory (VMI)**: Some customers have agreements (VMI), limiting which plants
  can
  fulfill their orders.

## Data Acquisition:

Using the dataset from
[Kaggle - Supply Chain Data](https://www.kaggle.com/datasets/laurinbrechter/supply-chain-data/data),
this project will involve data processing and model optimization to meet these requirements.

## Guidelines:

### Variables:

Define binary variables for:

- Order Assignment: Represents whether an order is processed by a specific plant.
- Port Selection: Indicates if an order is shipped through a particular port.

### Objective:

Minimize the total supply chain cost, which includes:

- Processing Costs: Associated with each plant handling an order.
- Shipping Costs: Based on the route from the selected port to the customer destination.

### Constraints:

* **Product Compatibility**: Ensure orders are only assigned to plants capable of processing the
  specific product type.
* **Port Accessibility**: Plants can only use ports they are connected to for shipment.
* **Vendor Managed Inventory (VMI)**: Certain customers are limited to specific plants for order
  fulfillment.

## Collaboration and Assessment:

### Phase 1: Data Preprocessing

Work collaboratively to clean and standardize the dataset, ensuring compatibility for modeling in
Gurobi or other MIP solvers. Maintain consistency across variables and constraints through group
collaboration.

### Phase 2: Model Formulation

Divide into groups (Gomory & Dantzig) to develop a mathematical model addressing the objectives and
constraints. Each group should produce:

- Mathematical Formulation: Clear articulation of the MIP model, including objective function and
  constraints.
- Technical Report: A concise document detailing your model design, decision variables, objective
  function, and constraints.

### Software & Participation:

Use Gurobi for solving; you might need to secure a free academic license, in case your model is
too large for the default free-tier.
Active participation in the modeling is paramount. Engage in constructive discussions regarding the
model in the GitHub issues list.

### Presentation & Evaluation:

You will present your collective findings in class on November 7th 2024. Each member must make
their presence felt, as I will undertake an oral assessment to gauge their comprehension of the
task. The essence of this exercise is not for members to fragment the work and tackle their
segments. Although roles might be designated to specific members, comprehensive understanding is
pivotal. Each one must be familiar with the team's undertakings, and no one should be left in the
dark.

In times of ambiguity, seek clarification, or for additional insights, do not hesitate to tag me on
GitHub,
`@tungufoss` or via e-mail at `helgaingim@hi.is`. My office is located in room 254a in VR-II.

## Submission:

Your MIP formulation, presented in an articulate manner, should be submitted. This report should
meticulously detail your optimal supply chain allocation, conforming to the constraints
described above.

Submit your report in PDF format to this GitHub repository, and send the URL to the file via the
Canvas assignment.
The deadline for submission is November 7th 2024.

This project is designed to simulate real-world supply chain optimization challenges. Good luck and
enjoy refining your MIP modeling skills!






