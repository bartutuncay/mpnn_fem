# Meeting Notes - December 1

## From last week's presentation 
- What I should focus on:
    - Presentation (smooth out)
    - Clear direction


## From last week
- Simpler dataset with I-beams:
    - Single material
    - Varying cross-section
    - Varying loads
    - Identical boundary conditions
    - Analytical solution
    - ~800 samples currently
    - limit number of nodes

- Simpler implementation of the model
    - GENConv with edge features
    - Displacement as only output

## To do:
- Setting up baseline model with simpler dataset
- *Logging parameters one change at a time*
    - Test number of message passing steps
        - Plot steps vs. accuracy
    - Implement pooling from GABI repository
        - Node features summarized from graph
- Investigate pooling
    - NeurIPS paper
    - Graph pooling & graph modes
    - Condensation

## Further steps:
- Investigate loss functions