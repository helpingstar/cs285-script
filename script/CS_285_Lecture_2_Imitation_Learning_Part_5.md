1. All right, the last topic we're going to talk about is the Dagger algorithm.
2. And the Dagger algorithm is actually something that you're going to be implementing in your homework.
3. And the Dagger algorithm aims to provide a more principled solution to the imitation learning distributional shift problem.
4. So as a reminder, the problem with distributional shift intuitively is that your policy makes at least small mistakes, even close to the training data.
5. And when it makes small mistakes, it finds itself in states that are more unfamiliar.
6. And there it makes bigger mistakes, and the mistakes compound.
7. More precisely, the problem can be described as a problem of distributional shift, meaning the distribution of states under which the policy is trained, pdata, is systematically different from the distribution of states under which it's tested, which is ppi θ.
8. And so far, a lot of what we talked about are methods that try to change the policy so that ppi θ will stay closer to pdata by making fewer mistakes.
9. But can we go the other way around?
10. Can we instead change pdata?
11. So that pdata better covers the states that the policy actually visits?
12. Okay.
13. How can we make pdata be equal to ppi θ?
14. Well, of course, if we're changing our data set, we're introducing some additional assumptions.
15. So we're going to be actually collecting more data than just the initial demonstrations.
16. And the question then is which data to collect.
17. And that's what Dagger tries to answer.
18. So instead of being clever about ppi θ or about how we train our policy, let's be clever about our data collection strategy.
19. So the idea in Dagger...
20. is to actually run the policy in the real world, see which states it visits, and ask humans to label those states.
21. So the goal is to collect data in such a way that ppi θ...
22. that the trained data comes from ppi θ instead of pdata.
23. And we're going to do that by actually running our policy.
24. So here's the algorithm.
25. Now, we're going to need labels for all those states.
26. We're going to train our policy first on our training data, just on our demonstrations, to get it started.
27. And then we'll run our policy and we'll record the observations that the policy sees.
28. And then we'll ask a person to go through all of those observations and label them with the action that they would have taken.
29. And now we have a labeled version of the policy data set.
30. And then we're going to aggregate.
31. We're going to take the union of the original data set and this additional label data set that we just got.
32. And then go back to step one, retrain the policy and repeat.
33. So every time through this loop, we run our policy, we collect observations.
34. We ask humans to label them with the correct actions for those observations.
35. And then we aggregate.
36. And it can actually be shown that eventually this algorithm will converge such that eventually the distribution of observations in this data set will approach the distribution of observations that the policy actually sees when it runs.
37. The intuition for why that's true, of course, is that eventually, is that each time the policy runs, you collect its observations, but then you might label them with actions that are different from the actions it took.
38. But that distribution is closer than the initial one.
39. So as long as you get closer each step, eventually you'll get to a distribution where the policy can actually learn and then you'll stay there forever.
40. So then as you collect from it more and more, eventually your data set becomes dominated by samples from the correct p π_θ distribution.
41. So that's the algorithm.
42. It's a very simple algorithm to implement if you can get those labels.
43. Here's a video of this algorithm in action.
44. This is in the original Dagger paper.
45. This was a about 12 years ago where they actually used it to fly a drone through a forest and Dagger was used to where they actually flew the drone, collected the data and then asked humans to label it offline by actually looking at the images and using a little mouse interface to specify what the action should have been.
46. And with a few iterations of Dagger, they can actually get it to fly pretty reliably through a forest, dodging trees.
47. Now, there is, of course, a problem with this method, and that has to do with step three.
48. It's sometimes not very natural to ask a human to examine images after the fact and output the correct action.
49. When you're driving a car, you're not just instantaneously making a decision every time step about which action to choose.
50. You are situated in a temporal process.
51. You have reaction times, all that stuff.
52. So sometimes the human labels that you can get offline in this sort of a counterfactual way can be not as natural as what a human might do when they were actually operating the system.
53. So step three can be a bit of a problem for Dagger and many improvements on Dagger seek to alleviate that challenge.
54. But the basic version of Dagger works like this, and that's the version that you will all be implementing in your homework.
55. There's really not much more to say about Dagger.
56. It alleviates the distributional shift problem.
57. It actually provably addresses it.
58. So you can derive a bound for Dagger and that bound is linear in T rather than quadratic.
59. But of course, that comes at the cost of introducing this much stronger assumption that you can collect the additional data.
60. OK, so that's basically the list of the methods I wanted to cover for how to address the challenges of behavior cloning.
61. We can be smart about how we collect and augment our data.
62. We can use powerful models that make very few mistakes.
63. We can use multitask learning or we can change the data collection procedure and use Dagger.
64. The last thing I want to mention, which is a little bit of a preview of what's going to come next, is why is imitation learning not enough by itself?
65. Why do we even need the rest of the course?
66. Well, humans need to provide data for imitation learning, which is sometimes fine, but deep learning works best when the data is very plentiful.
67. So asking humans to provide huge amounts of data.
68. Can be a huge limitation.
69. If the if the algorithm can collect data autonomously, then we can be in that regime where DeepNets really thrive and data is very plentiful without exorbitant amounts of human effort.
70. The other thing is that humans are not good at providing some kinds of actions.
71. So humans might be pretty good at specifying whether you should go left or right on a hiking trail or controlling a quadcopter through a remote control.
72. But they might not be so good at, for example, controlling the low level commands to quadcopter rotors to make it do some really complex aerobatic trick.
73. If you want humans to control all the joints in a complex humanoid robot, that might be even harder.
74. Maybe you need to rig up some really complicated harness for them to wear.
75. If you want to control a giant robotic spider, well, good luck finding a human who can operate that.
76. And humans can learn things autonomously and just intellectually, it seems very appealing to try to develop methods that can allow our machines to do the same.
77. As I mentioned in lecture one, one of the most exciting things we can get out of learning based control is emerging behaviors, behaviors that are better than what humans would have done.
78. And in that case, it's very desirable to learn autonomously.
79. When learning autonomously, in principle, machines can get unlimited data from their own experience, and they can continuously self-improve and get better and better, in principle, exceeding the performance of humans.
80. Now, in order to start thinking about that, we have to introduce some terminology and notation.
81. We have to actually define what it is that we want.
82. If our goal is no longer just to imitate, but we want to do something else, well, what is it that we want?
83. And maybe instead of matching the actions in the expert data set, we want to bring about some desired outcome.
84. Maybe in the tiger example, we want to minimize the probability of being eaten by the tiger.
85. So, we want to minimize the probability that we will land in a state, s', which is an eaten by tiger state.
86. And we can write that down mathematically.
87. And in general, we can write it as the expected value of some cost.
88. In this case, the cost is being eaten by a tiger.
89. Now, we already saw costs before when we talked about counting the number of mistakes.
90. But in general, we can have arbitrary costs on states and actions.
91. And those can define arbitrary control tasks, like not being eaten by tigers or reaching a desired destination.
92. So, the new thing that we're going to introduce and that we're going to use in lectures next week is the cost function.
93. Or, sometimes, the reward function.
94. Now, the cost function and the reward function are really the same thing.
95. They're just negatives of one another.
96. And the reason that we see both sometimes is the same kind of a cultural distinction that I alluded to before.
97. Remember, I mentioned that we have S and A, which comes from the study of dynamic programming.
98. And that's where the reward comes from.
99. In optimal control, it's a bit more common to deal with costs.
100. I don't know if there's a cultural commentary here.
101. Well, you know, optimal control originated in Russia.
102. Maybe it's a little more common to think about costs in America.
103. We are all very optimistic, and we think about life as bringing rewards.
104. Maybe there's something to that.
105. But for the purpose of this class, don't worry about it.
106. C is just a negative of R.
107. And to bring this all the way back around to imitation, well, the cost function that we saw before for imitation can be framed in exactly the same framework.
108. We have rewards, which are log probabilities.
109. We have costs, and those are interchangeable.
110. You can have the cost be the negative of the reward, and you can define a cost for imitation.
111. But you can define a more expressive cost for the thing you actually want, like reaching your destination or avoiding a car accident, and then use those with the more powerful reinforcement learning algorithms that we'll cover in future weeks.