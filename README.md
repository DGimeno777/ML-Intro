# ML-Intro
Repo for learning ML basics

## Resources
I'm using a beginner's guide [here](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/) to get a grasp of the basics of ML and create a practical first project. In addition to following the guide I'll be updating the notes section of this doc with any thoughts or tricks.

### Readings/Helpful Links:
- [Linear vs. Non-Linear ML Algs](https://www.quora.com/Whats-the-difference-between-linear-and-non-linear-machine-learning-model)
- [Confusion Matrix](https://machinelearningmastery.com/confusion-matrix-machine-learning/)
-

## Notes
- Use Pandas's .describe() function to summarize dataset into count, mean, percentiles, and other stats
- Noticed it wasn't instantaneous to do the model training so I added some basic timestamp difference to the loop to see but it seems like each algorithm (except for CART) took the same amount of time to be processed
- I assumed from the text that it would take longer to train the non-linear algorithms but it could be that our dataset is either too small or not complicated enough (e.g. not something like image recognition) to see any meaningful distance