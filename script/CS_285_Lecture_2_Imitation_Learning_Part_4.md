1. All right, in part four of the lecture, I'll talk about a few algorithmic approaches that can also make behavioral cloning work pretty well.
2. The first one I'll discuss is to use multitask learning.
3. So this might seem a little paradoxical at first, but it turns out that sometimes learning many tasks at the same time can actually make imitation learning easier.
4. So let's say that you would like to train your agent, let's say your vehicle, to drive to a particular location.
5. Let's call it location P1.
6. And you might have lots of demonstrations of driving to location P1.
7. And then you'll train a policy A given S.
8. So that's a pretty straightforward thing to do.
9. But as we discussed before, if you want to train a policy like this to be robust so that it doesn't suffer too much from compounding errors, maybe you would really want to get the expert to make some intentional mistakes, put it into some states where it can recover from those mistakes and teach the policy that way.
10. Well, what if you instead don't have such optimal expert data?
11. Instead, you have data of the expert attempting to drive to lots of different locations.
12. What you can do is you can actually train a policy that receives the desired location as input.
13. And the way that you get the desired location is by just looking at the last time step that the human expert landed into.
14. And then you'll train a policy for reaching any P.
15. One of the nice things about this is that, of course, the expert will visit many more different states if they're trying to go to many different locations.
16. So by conditioning the policy on the location, you can still get a policy for the location P1 that you wanted, but you're getting a lot more training data.
17. And perhaps more importantly, you're getting data from lots of different states that the expert might not have visited if they were just trying to reach P1 and if they were behaving optimally.
18. So what you can do is...
19. Something called goal-conditioned behavior cloning.
20. At training time, you might receive a collection of trajectories where you're not even told what the expert is trying to do.
21. They're just sequences of states and actions.
22. And you assume that whatever the expert was doing was a good example for the state that they actually reached.
23. So you say that, well, demo one is a good demo for reaching the state S capital T.
24. Demo two is a good demo for reaching whatever state that reached.
25. And then you just feed in the last state as an additional input into the policy and train the policy to take the action that the expert took whenver the expert trying to reach that state.
26. And that gives you access to a lot more training states that provide much better coverage and hopefully give you many more of those instances where you might learn corrections.
27. So in this case, while you still suffer from distributional shift, you might still make mistakes and find yourself in states that are unfamiliar for the particular goal you're commanding.
28. That state might be more familiar for some other goal.
29. And the other really nice thing about this is that you can actually leverage data that is more suboptimal.
30. Because maybe...
31. The expert failed at reaching the position P1, but they succeeded at reaching some other position, and you can still learn from that.
32. So for each demo, you maximize the log probability of the action in the demo, given the state and given the last state in the demo.
33. That's basically the entirety of the method.
34. And this is goal-conditioned behavioral planning.
35. So you just feed in two states instead of one.
36. Now, one thing I will note here is that while in practice, this often makes things work better.
37. In theory, this methodology is actually a little bit problematic because now we actually see distributional shift in two places.
38. We see distributional shift as before in the sense that our state distribution is different.
39. Our P train is different from...
40. Sorry, our P data is different from P π_θ.
41. But we also see distributional shift in another place when we do relabeling like this.
42. And I'll leave that part as an exercise to the reader and something we could discuss in class.
43. So as a hint, we see distributional shift actually in two places.
44. So we're in train this way.
45. And you could think about what that second source of distributional shift is.
46. So in theory, this is actually potentially worse.
47. But in practice, it's often better.
48. So let me show you a few examples of works that have done this.
49. The goal-conditioned behavioral cloning method was arguably popularized by these two papers, Learning Latent Plans for Play and Unsupervised Visio-Motor Control through Distributional Planning Networks.
50. And I'll talk about learning latent plans from play a little bit first.
51. So the concept there was to collect data with humans that were not actually told to do any tasks in particular but they were just told to play around with objects in their environment so in this dataset the people are using a VR controller to control the simulated robot and they're kind of performing random stuff so they're not moving the arm random either but they're performing random tasks and that of course covers lots of different states so a policy train on this kind of data would have a really hard time trying to find a state that is @ of distribution because almost all the reasonable states have actually been seen here some where or at least stats are very much like them.
52. But of course it's not clear what task is being performed in each of the trajectories.
53. So by taking this data and performing this goal relabeling, where every trajectory is labeled with the state that was actually reached later on in that trajectory, and using a latent variable policy that can express multimodality, the authors of this work were actually able to get a pretty effective policy for reaching a wide variety of goals.
54. So this uses that latent variable model from before, and it uses the goal relabeling, and putting them together you can get a policy where you can give it a goal, like a state where the door is closed or a state where the drawer is open, and the robot arm will actually autonomously go and do that.
55. So you can see that it actually does a pretty significant variety of behaviors all in a single policy.
56. One of the interesting things you could do with these goal condition behavior cloning methods is you can actually use them as online self-improvement methods, very similar in spirit to RL.
57. So these are not, I guess, true RL methods, but they are RL-like in that they can improve through experience.
58. So the idea is that you can start with a random policy, collect data by commanding the policy to go to random goals, treat this data as demonstrations for the state that was actually reached, so relabel them for the state that these random trajectories reached, use that to improve the policy, and then run it again.
59. And the idea is that initially the policy does mostly random things, but then it learns about the actions that led to the states that it actually reached, and then it can be more deliberate on the next iteration.
60. So the method simply applies this goal relabeling imitation learning approach iteratively running relabeling, imitation, then more data collection, then more relabeling, and then more imitation.
61. And that can actually be a pretty decent and simple way to improve a policy.
62. The other nice thing about these goal condition behavior cloning methods is that they're quite scalable.
63. So you can apply them at a huge scale.
64. This next case study I'm going to tell you about, this was a paper led by Dhruv Shah Ajay Sridor they did is they developed a policy for driving ground robots, not autonomous cars yet, but smaller scale ground robots, that could actually generalize across many different kinds of robots.
65. So it's a goal conditioned imitation learning method that takes in the current observation and the goal image and actually takes in a history to deal with that non-Markovian-ness problem and then it outputs the action.
66. And it's trained on data collected from many different kinds of robots, ranging from small scale RC cars to full large scale ATVs.
67. And the cool thing about this policy is that it can then reach goals even for new types of robots that it was not trained in, like for example this drone in the top left corner of the video.
68. The policy was never trained on drones, but it can actually control drones in zero shot by generalizing to them from being trained on lots of different vehicles.
69. And you can see that it's using some of the ideas we discussed, it's of course using this goal relabeling trick and it's using a history that is read in, in this case, by just concatenating the frames, although in later work it's also read in with a sequence model transformer.
70. The last thing I want to mention here is a paper called hindsight experience replay, which introduced a very similar principle but in the context of off-policy reinforcement learning algorithms.
71. We'll talk about off-policy reinforcement learning much more later, I didn't describe what this is yet, but I just wanted to mention this paper because it is something that often comes up in the context of this work.
72. It is not doing goal conditioned behavior cloning, but it is applying a hindsight relabeling method to off-policy RL and actor critic methods.
73. So we'll talk about off-policy RL and we'll talk about actor critic methods later, but I want to mention this because it is an idea that's also very widely used in current methods.