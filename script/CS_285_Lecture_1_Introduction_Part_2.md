1. Okay, let's talk about what we'll cover in the class.
2. So this course goes through a variety of deep reinforcement learning methods construed very broadly.
3. We'll start with some basics.
4. We'll start by talking about how we can take a journey from supervised learning methods to decision-making methods, provide some definitions, and generally come to understand the reinforcement learning problem.
5. Then we'll have a unit on model-free reinforcement learning algorithms, where we'll cover Q-learning, policy gradient, and actual critic methods.
6. And you'll have some homeworks where you'll implement each of these.
7. Then we'll have another unit on model-based algorithms.
8. We'll talk about planning, optimal control, sequence models, images, and things like that.
9. And then we'll have a variety of more advanced topics.
10. We're going to cover exploration algorithms.
11. We're going to cover algorithms for offline reinforcement learning, which are methods that can use both data and reinforcement learning methods.
12. We'll talk about inverse reinforcement learning, which deals with inferring objective functions from behavior, and have some discussion in there about the relationship between reinforcement learning methods and things like probabilistic inference.
13. And then we'll have a few advanced topics like meta-learning and transfer learning, maybe hierarchical RL, and a set of research talks and invited lectures.
14. So that's the overall overview of the class.
15. You're going to have five assignments.
16. There will be an assignment on imitation learning, policy gradients, Q-learning and actual critic algorithms, model-based RL, and the last one will be on offline RL.
17. And there will be a final project.
18. The final project.
19. The final project is a research-level project of your choice.
20. You can form a group of up to two to three students, and you're more than welcome to start on this project early.
21. Students every year have some questions about our expectations for the scope of this project.
22. Roughly speaking, you should think about it as roughly at the level of a paper that you might submit, for example, to a workshop.
23. If you're not sure about the scope of your project, definitely come into office hours to talk to the TAs or to myself.
24. We will have multiple rounds of feedback.
25. We'll have a project proposal deadline and a project milestone report.
26. These are really meant for you.
27. We strongly encourage you to write up your plan, to describe potential concerns you have about your plan and so on, and we'll give you feedback on those.
28. So the proposal and the milestone are not graded very strictly.
29. They're really meant much more so for you to get feedback on your project plan before the final report at the end of the semester.
30. You'll be graded 50% on the homeworks, 40% on the project, and 10% on quizzes after every lecture.
32. And you'll have a total of five late days for your homework.
33. So don't exceed those five late days.
34. That's five late days total.
35. If you exceed them, then we unfortunately cannot give you credit for that homework.
36. You also have a little bit of homework for today.
37. Make sure that you're signed up for ED, UC Berkeley CS 285.
38. All of you who have been signed up for the course officially would have received an invitation for this.
39. We strongly encourage you to start forming project groups, unless you want to work alone, which is fine.
40. And take the Lecture 1 quiz.
41. So the Lecture 1 quiz is posted on Gradescope.
42. The Lecture 1 quiz is very much a practice quiz.
43. It's not a, there's not a real quiz there.
44. And it's really to get you to be familiar with the Gradescope interface.
45. However, what I want to focus on mainly in today's lecture is discussing why we should study reinforcement learning, what it is, and a little bit of context for why I myself like to teach this class.
46. But let's start with some basics.
47. What is reinforcement learning?
48. Well, reinforcement learning is really two things.
49. It's a mathematical formalism for learning-based decision-making.
50. And it's also an approach for learning decision-making and control from experience.
51. And it's important to keep in mind these are somewhat separate because we could imagine taking the formalism and then applying all sorts of different methods to it.
52. So it's important not to confuse reinforcement learning the problem from reinforcement learning the solution.
53. Okay, how is this different from other machine learning topics?
54. Well, the kind of machine learning that most of you are probably familiar with is supervised learning.
55. Supervised learning is fairly straightforward to define.
57. You have a data set of inputs and outputs.
58. We refer to them typically as x and y.
59. And you want to learn to predict y from x.
60. So you want to learn some kind of function, f of x, which outputs values y that are close to the y labels in the data set.
61. So, for example, f might be represented by a deep neural network that you would train via classification or regression to match the labels y.
62. And while the basic formulation of supervised machine learning is very straightforward, supervised machine learning methods make a number of assumptions that we often don't even think about because they're so natural, but that are important to bring up if we're going to discuss how this differs from reinforcement learning.
63. Supervised learning typically assumes what is called independent and identically distributed data.
64. This is such an obvious assumption in some ways, especially for someone who has studied machine learning, that we often don't make it explicit.
65. But what it means is that all of these x, y pairs in your data set are independent of one another.
66. In the sense that the label for one x doesn't influence the label for another x.
67. And they're distributed identically in the sense that the true function that produced the label y from x is the same for all the samples.
68. It's almost an obvious statement, but it's something that is important to keep in mind.
69. Supervised learning also assumes that our data set is labeled in the sense that every x we've seen in D also has an accompanying y, and that y is the true label for that x.
70. This is very natural if you're doing things like image classification with labels obtained from humans.
71. But remember how we discussed in the grasping example, this can actually be pretty unnatural.
72. If you want a robot to learn how to grasp objects, it's actually a very strong assumption to assume that you're given a set of images with ground truth optimal grasp locations.
73. Reinforcement learning does not assume that the data is independent and identically distributed, in the sense that previous outputs influence future inputs.
74. Things are arranged in a temporal sequence, and the past influences the future.
75. Typically, the ground truth answer is not known.
76. It's only known how good a particular outcome was, whether it was a failure or a success, or more generally what its reward value was.
77. So in reinforcement learning, you might collect data, but you can't simply copy that data.
78. That doesn't actually lead to success.
79. The data might tell you which things were successful and which things failed, although even those labels are difficult to interpret properly, because if you have a sequence of events that led to a failure, you don't know which event, which particular choice it was, and which case in that sequence was the one that precipitated the failure.
80. This is not unlike human decision making.
81. Perhaps you got a really bad grade at the end of a course.
82. Well, it wasn't the fact that you looked up your grade on CalCentral that caused you to get the bad grade.
83. It was something you did earlier in the class, perhaps the fact that you did poorly on an exam.
84. At the time, perhaps you didn't realize that this would lead you to fail the course.
85. So this is very much an issue that we have in reinforcement learning, referred to as credit assignment, where the decision that actually results in a bad outcome, or a good outcome, might not itself be labeled with a high or low reward.
86. The reward might only happen later.
87. So we need to take this data, which is not labeled with ground truth optimal outputs, and might involve these delayed rewards, run reinforcement learning on it, and hopefully get a behavior that is better than the behavior we saw before.
88. So that's really the challenge in reinforcement learning.
89. So let's try to make this a little bit more precise.
90. In supervised learning, you have an input x and you have an output y, and you have a data set that consists of x-y pairs.
91. The goal is to learn a function that takes an x and approximates y.
92. And typically this function has some parameters which are referred to as theta.
93. These might be, for example, the weights in a neural network.
94. In reinforcement learning, we have a kind of a cyclical online learning procedure where an agent interacts with the world, the agent chooses actions, at, at every point in time, and the world responds with the resulting state, st plus one, and a reward signal.
95. And the reward signal simply indicates how good that state is, but it doesn't necessarily tell you if the action that you just took was a good or bad action.
96. Perhaps you got lucky and landed in a good state, or perhaps you did something really good earlier that caused you to get into a good state now.
97. The input to our agent is going to be the state st at each time step.
98. So this is kind of the n log of x.
99. The output is at at each time step.
100. The data, which is collected by the agent itself classically, consists of sequences of states, actions, and variables.
101. Rewards are numbers, scalar values.
102. And whereas in supervised learning, the data is given to you, you don't have to worry about who gave you the data, it's just provided to you as a set of x, y tuples, in reinforcement learning, you have to pick your own actions and collect your own data.
103. So not only do you have to worry about the fact that the actions in your data set might not be the optimal actions, you have to also actually decide how that data will be collected.
104. And your goal is to learn a policy, pi theta, which maps states s to actions a.
106. And just like f, pi has parameters theta, so those might again be the weights in a neural network.
107. And a good policy is one that maximizes the cumulative total reward.
108. So not just the reward at any point in time, but the total reward the agent receives.
109. So that involves strategic reasoning.
110. Maybe you might do something that might seem unrewarding now to attain higher rewards later.
111. So let's talk about some examples of how problems could be cast in the terminology of reinforcement learning.
112. Let's say that you'd like to train a dog to perform some trick.
113. So in this case, the actions might be the muscle contractions of the dog's muscles.
114. The observations might be what the dog perceives through its sense of sight and smell.
115. The reward might be the treat that it gets.
116. And the dog will then learn to do whatever maximizes that reward, which might be the trick that you want to perform because you reward it with food when the dog performs the trick successfully.
117. Okay?
118. Here's another example.
119. Maybe you have a robot.
120. Its actions might be the motor current or torque, some kind of actuation command sent to its motors.
121. Its observations might be the readings from its sensors, like camera images.
122. And its reward might be some measure of task success.
123. Maybe this robot needs to run as fast as possible to reach a destination, so its reward function might be the running speed, or it might be whether it reached the destination or not.
124. Maybe it receives a plus one when it reaches the destination successfully, and a minus one otherwise.
125. Here's another problem.
126. Let's say you want to manage inventory, route goods between different warehouses, in order to maintain stock levels.
127. Perhaps the actions are which inventory to purchase.
128. The observations are the current inventory levels, and the reward might be the profit you make.
129. Perhaps you have to pay if you want to store inventory for a long time, so your profits will be low.
130. So you can see that this formulation is very general.
131. Many different problems can be cast into the framework of reinforcement learning.
132. We'll of course make all of this a lot more precise later, so this is a very high-level introduction.
133. Don't worry yet if the particular details aren't very clear.
134. We'll make this a lot more precise in later lectures.
135. But for now, let me just give you some examples of the kinds of things that reinforcement learning methods could do.
136. One of the things that reinforcement learning is very good at is learning policies for physically complex tasks, tasks where it might be very difficult for a person to describe precisely how the task should be performed, but much easier to define the reward.
137. Like in this case, the reward is that the nail should be hammered in, and the reinforcement learning algorithm figures out how to control this robotic hand to move the hammer to hammer in the nail.
138. Here's another complex physical task.
139. Here, this quadrupedal robot needs to be able to jump over different obstacles.
140. Now, coding up manually a skill for jumping like this is very tough, but reinforcement learning can learn the actuations that will allow the robot to jump in different locations, to various distances, and so on.
141. It can even perform more physically complex tasks, so here in the next clip, the quadrupedal robot, this is a baseline method, so don't worry about this.
142. Here, this quadrupedal robot needs to figure out how to stand on its hind legs and balance, and that's also very difficult to do manually, but with an appropriate reinforcement learning method, that's actually possible.
143. In fact, reinforcement learning has been applied very widely to robotic problems.
144. Here's an even more recent work, and this is from ETH Zurich, showing a robot using reinforcement learning with a combination of simulated optimization, to learn various agile skills.
145. And you can see that it can climb onto obstacles and things like that.
146. The other thing that reinforcement learning is great at is coming up with unexpected solutions.
147. I alluded to this before with the AlphaGo example.
148. Here's another example, which you'll actually implement in your homeworks, where a Q-learning algorithm is learned to play Atari, and discovered this strategy that if you bounce the ball up over the bricks, then it will bounce around, and you'll get lots of points.
149. Reinforcement learning can also be applied at larger scale in the real world.
150. This is a project, that was done at a company called Everyday Robots, which is an alphabet company, where the robots learn to sort trash.
151. So the idea is that if people put trash into recyclables, that should actually go into the compost, then the robot can come and sort it.
152. The robots here learn in the real world, both in these classroom environments where they can practice, and in actual office buildings, and these vision-based skills that are kind of similar in spirit to the ones that I mentioned in the beginning.
153. Can then pick up and move objects in real world office buildings.
154. So that's pretty neat that you can actually practice these things.
155. You can practice them on the job, you can practice them in the real world.
156. And it's of course not just for games and robots.
157. I really like this next example.
158. This is work that was done by Cathy Wu, who's now a professor at MIT, and was previously a PhD student here at UC Berkeley.
159. And what Cathy was working on is reinforcement learning algorithms for controlling traffic.
160. And this is a kind of a toy example where these cars drive in a circle, and what tends to happen, even in a simple circular environment like this, if you have a very accurate model of how human drivers behave, you'll actually get traffic jams forming spontaneously.
161. So cars will kind of bunch up, and when they bunch up like this, they'll actually spontaneously form traffic jams, even though they drive in a circle.
162. So what Cathy then did is she optimized the reinforcement learning policy, which will be shown next for the car shown in red, to not optimize its own speed, but to optimize the speed of the entire circle.
163. And you can see that what this car in red is going to do is it's going to actually slow down and wait for everybody to resolve the traffic jam, and by going a little bit slower, it'll actually avoid the formation of traffic jams in the entire circle.
164. Cathy also experimented with this in other settings.
165. This is a figure-eight kind of intersection, and as you might expect, cars will bunch up at this intersection and cause delays.
166. So if there's an autonomous car that is trying to optimize the driving speed of all the cars, the autonomous car will actually slow down a little bit and regulate the traffic so that everyone passes through the intersection at exactly the perfect time.
167. Now, this example maybe is a little bit synthetic, but there's a considerable follow-up work to this showing that, in fact, autonomous regulation of traffic with reinforcement learning can be quite a powerful tool.
168. Reinforcement learning has also been used very widely with language models.
169. Many of you are probably familiar with the advances in recent language models with things like ChatGPT and many other systems like Anthropix Cloud or Google's Bard, which use large amounts of data to train models that will fulfill user requests.
170. And this is an example on the right where someone asked ChatGPT to explain how RL with human feedback works for language models, and it produces some kind of explanation.
171. Now, by themselves, large language models trained on lots of internet data can solve very sophisticated problems, but it's quite difficult to persuade them to do this because these models are basically trying to complete text based on what they learn from internet data, so you have to prompt them in a way that kind of indexes into the right context.
172. Reinforcement learning can be used to make this a lot easier by essentially training these models based on human scores.
173. So instead of just asking them to provide the kind of completions that are most likely from internet data, they can actually be trained to respond to queries in ways that human raters find to be desirable.
174. And reinforcement learning is actually a very important part of this.
175. Reinforcement learning has also been used with image generation.
176. Here's an example with Stable Diffusion 1.4.
177. If you ask it to generate a picture of a dolphin riding a bike, it actually generates a picture that is not very good for this.
178. What you can do is you can take this image and you can give it to a captioning model, in this case, LLaVA, to produce a description of the image, and then use RL, where the reward function is given by the similarity between the description from LLaVA and the original prompt.
179. So when Lava looks at this picture, it might say, oh, this is a picture of a dolphin above the water, which is not very similar to a dolphin riding a bike, so it receives a bad reward for that.
180. If we then optimize the image generation model with RL to maximize this reward, it'll gradually make the image more appropriate to the prompt.
181. So now there's both a dolphin and a bicycle, although the dolphin's not riding the bicycle just yet.
182. With a few more iterations, now there's a dolphin like creature that is, in fact, on a bicycle.
183. And with some more iterations, the creature becomes much more clearly a dolphin, apparently putting some waves in the background makes it extra dolphin-like, and then eventually there's a full-fledged picture of a dolphin riding a bicycle.
184. So reinforcement learning can be used to optimize image generation models.
185. Reinforcement learning can also be used for other things.
186. This is an example on chip design, where the actions correspond to placement of chip parts for layout, and the reward has to do with various chip design parameters like the cost or the congestion or the latency of the chip.
187. So reinforcement learning can actually be applied quite broadly.
188. So I'll pause here, and in the next section I'll discuss why we should study Deep RL today.