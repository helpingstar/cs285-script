1.  All right, next let's get started discussing some reinforcement learning algorithms.
2. There are quite a few reinforcement learning algorithms that we'll cover in this course, but at a high level, all of these algorithms will share more or less the same high-level anatomy.
3. They will consist of three basic parts.
4. The first part, which I will always draw in orange, is to generate samples.
5. So reinforcement learning is about learning through trial and error in some sense.
6. And the trial part of trial and error means actually attempting to run your policy in your environment, actually have it interact with your markup decision process, and collect samples.
7. What are samples?
8. Well, samples are trajectories.
9. So you will interact with your MDP, and typically what you will be doing is you will be sampling trajectories from the trajectory distribution, typically the one induced by your policy, although when we talk about exploration later, sometimes you might actually choose to sample from a slightly different trajectory distribution than the one that our policy would define.
10. But for now, let's just assume that generating samples means sampling trajectories from the trajectory distribution defined by your policy, which simply means running your policy in the environment.
11. Then we will have the green box, and the green box corresponds to learning some kind of model.
12. This could be...
13. literally a model of the dynamics in a model-based RL algorithm, or it can be some sort of more implicit model such as a value function.
14. And this green box basically corresponds to estimating something about your current policy, something about how your policy is doing, how well it's performing, what kind of rewards it's attaining.
15. And then we'll have the blue box, which is where you actually change your policy to make it better.
16. And then you repeat this process.
17. Pretty much all of the algorithms that we will cover will have these three parts.
18. In some cases, one of these parts might be very, very simple.
19. In some cases, some of them might be very complex.
20. But all three of them are generally going to be present.
21. So here are some simple examples.
22. Let's say that we're going to run our policy and we'll generate sample trajectories denoted by these black lines and we'll evaluate them to see if they're good or bad.
23. So evaluate means just sum up their rewards.
24. Summing up their rewards is what happens in the green box.
25. And then when we improve the policy, we might try to make the good trajectories, kind of the green check marks, be more likely, and the bad trajectories, the red ones, be less likely.
26. So that's the improvement step.
27. Now what I'm describing here is the basic high-level scheme for a policy gradient algorithm, which we'll learn about in detail in the next lecture.
28. So in a policy gradient algorithm, the green box, the box where we estimate something about our policy, is very, very simple.
29. It simply consists of summing up the rewards along the trajectories that we sampled, and that tells us how good our policy is.
30. So the green box is just a summation.
31. The blue box might involve calculating the gradient of the reward of your policy, and we'll talk about how to do that in the next lecture, and applying that gradient to the policy parameters, theta.
32. So that's a very simple trial and error style.
33. Reinforce the learning algorithm.
34. Run your policy, get some trajectories, measure how good those trajectories are, and then modify the policy to make the better trajectories have a higher probability.
35. You could also imagine doing a model-based RL procedure.
36. You can kind of think of this as RL by back-prop.
37. So maybe in the green box, you learn a model, you learn some other neural network, F phi, such that S , is approximately equal to F phi of S , and you train F phi on data generated in the orange box with supervised learning.
38. So maybe you have a whole other neural network that goes from S to S .
39. Now this green box is much more complex than the one we had on the previous slide.
40. On the previous slide, we would just sum over the rewards in our trajectories.
41. Here, we're actually fitting a whole other neural network.
42. The summation might take a millisecond.
43. This might take minutes to train, or maybe even hours if it's using images.
44. And then in the blue box, we might back-prop through F and R to train the policy pi theta.
45. So if we want to calculate the reward of the policy, we can basically compose the policy with F, actually use our automatic differentiation software to calculate the reward, and back-prop through all of them to optimize the policy.
46. We're going to cover methods that do some variant of this when we talk about model-based reinforcement learning much later in the course.
47. If the details of this don't currently make sense, don't worry about it.
48. The only point of this slide is to explain to you the different incarnations of the green box and the blue box that we might see in some very different reinforcement learning algorithms.
49. All right.
50. Now, which parts in this whole process are expensive, and which parts might be cheap?
51. Well, the orange box, its cost in terms of time and computation depends a great deal on what kind of problem you're asking.
52. So the green box is the cost of the data you're solving.
53. If you're collecting samples by running a real-world system like a real robot, a real car, a real power grid, a real chemical plant, whatever, the orange box can be potentially extremely expensive, right?
54. Because you have to collect data in real time, at least until we invent time travel.
55. And if you need, like, thousands of samples each iteration of your RL algorithm, this can be very, very costly.
56. On the other hand, if you're collecting samples in the MuJoCo simulator that all of you are using for homework one, then, you know, the MuJoCo simulator can run at up to 10,000x real time, so the cost of the orange box might actually be trivial.
57. So depending on which of these regimes you're in or where you are on the spectrum, you might care more or less about how many samples you need in the orange box, which will influence your choice of reinforcement learning algorithms.
58. So this can range from prohibitively expensive to trivially cheap, depending on how you're learning.
59. The green box also could range from extremely cheap to extremely cheap.
60. So if you're just estimating the return of your policy by summing up the rewards that you obtained, this is very, very cheap.
61. It's just a summation operator.
62. If you are learning an entire model by training a whole other neural net, this might be very expensive.
63. It might require, you know, a big supervised learning run in the inner loop of your RL algorithm.
64. Similarly, in the blue box, if you're just taking one gradient step, this might be fairly cheap.
65. If you have to backprop through the blue box, you might want to go through your model and your policy like I discussed in the model-based slide.
66. This might be very expensive.
67. And there will be algorithms that fall at different points of the spectrum for different boxes.
68. For instance, a Q-learning algorithm, which we'll cover a couple weeks from now, basically spends all of its effort in the green box, and the blue box is just an ARC max.