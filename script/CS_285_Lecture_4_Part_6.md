1.  In the last portion of today's lecture, I'm going to just briefly go through a few examples of actual deep RL algorithms, just to kind of show you some of the things that they do.
2. This will be the least technical portion of the lecture, and these algorithms will be covered in much more detail in the subsequent lectures, whereas this part is mainly just to kind of round out today's lecture with some nice interesting examples and videos.
3. So some examples of specific algorithms, and don't worry if you haven't heard these names, we'll cover these more later.
4. Value function fitting methods, so these are things like Q-learning, DQN, temporal difference learning, these are all value function methods, fitted value iteration.
5. Policy gradient methods, these are methods like reinforce, natural gradient, transfusion policy optimization, or TRPO, PPO, etc.
6. Actor-critic algorithms, these are things like asynchronous advantage actor-critic or A3C, soft actor-critic, DDPG and so on.
7. Model-based RL algorithms, these are things like dyna, guided-transmission algorithms, and so on.
8. And we'll learn about most of these in the next few weeks.
9. But first, let's go through a few examples.
10. So here's a video of the Q-learning result for playing Atari games.
11. This is an algorithm that learns policies for playing video games directly from pixels.
12. This is from a paper by Miedal from 2013.
13. It runs around 26 D être, including 20.
14. If you're looking for value on astanden you can build 두 X value builds using X, but lots of people like to build six value builds with a D, while the system currently is in минут 5.
15. As you can see, this particular algorithm uses Q learning with convolutional neural networks.
16. So Q learning is a value-based method.
17. It actually learns an estimate of Q , S, A by using a neural network.
18. Atari games are discrete action environments, which means that you just have to produce a different cue value for each of small discrete set of actions, and then you take the art max over those Q values to select the best action while playing the game.
19. from the paper N10 Training of Deep Visumotor Policies.
20. And this is a model-based RL algorithm called Guided Policy Search, which uses a combination of dynamics models and image-based convolutional networks to perform a variety of robotic skills.
21. Here is a policy gradients example.
22. This is from the paper High Dimensional Continuous Control with Generalized Advantage Estimation.
23. It uses a variant of an algorithm called Trust Region Policy Optimization, which uses policy gradient methods.
24. A method that combines in this case a trust region with value function approximation.
25. So this is technically an actor-critic algorithm derived from policy gradient algorithm.
26. And here you can see it training this little humanoid robot how to walk.
27. And here is the video that I actually showed in the first lecture for the Grasping Robot.
28. This particular result was actually also produced by a Q-learning algorithm, not that different from the Atari learning algorithm, but it's a little bit more complex.
29. It's not that different from the Atari example that I showed a few slides ago, but in this case with a particular modification to handle continuous actions.