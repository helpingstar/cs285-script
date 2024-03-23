1. All right, the remainder of today's lecture will focus on more practical methods that can make behavioral cloning work, as well as some other algorithms that we could use.
2. So we talked about a little bit of theory, but now we'll talk about how the problem can be addressed in a few ways.
3. By being smart about collecting your data, by using very powerful models that make comparatively fewer mistakes, by using multitask learning, and by changing the algorithm.
4. And I'll go through these pretty fast, so my aim with this portion of the lecture is not really to go in great detail about how to actually implement some of these methods, but just to give you a sense for the types of methodologies that people employ.
5. The one method that you will implement in homework is Dagger, and I'll go through that somewhat more precisely.
6. Okay, so what makes behavioral cloning easy and what makes it hard?
7. As I mentioned in the previous part of the lecture, if you have very perfect data, then these accumulating errors are a big problem, because as soon as you make even a small mistake, you're outside of that distribution of perfect data.
8. But if you actually already have a bunch of mistakes in your data set, and corrections for those mistakes, then when you make a small mistake, you'll be in a state that is somewhat similar to other mistakes that you've seen in the data set, and the labels in that portion of the data set will tell you how to correct that mistake.
9. So there are a few ways that you could leverage this insight.
10. You could actually intentionally add mistakes and corrections during data collection.
11. That's not actually an entirely crazy idea.
12. So the mistakes will hurt, meaning that it will dilute the training set, but the corrections will help.
13. And often the corrections help more than the mistakes hurt.
14. The reason for it is that if the mistakes are somewhat random, they tend to average out, meaning that the most optimal action is still the most probable.
15. However, by making mistakes during data collection, you force the expert to provide examples in more diverse states, and that will teach the policy how to make corrections.
16. The simplest version of this you can think of is if you force the expert to make a mistake with some probability, independent of which state they're in, then the mistakes will be largely uncorrelated with the state, whereas the optimal action will be correlated with the state.
17. So when your neural network learns the action that correlates most of the state, it will actually tend to learn the optimal action and avoid the mistakes, but it will still benefit from seeing the corrections in those worse states.
18. Another thing that we could do is use some form of data augmentation.
19. And that camera trick from before can be thought of as a kind of data augmentation, essentially a method that adds some fake data that illustrates corrections, like those sight-facing cameras.
20. And that can be done with it by leveraging some sort of domain knowledge about the problem you're solving to create some additional fake data.
21. And roughly speaking, the effect of these two tricks is kind of the same.
22. In both cases, the aim is to provide examples in states that the expert is unlikely to visit, but that the policy might end up landing in.
23. Now, there isn't really much more to this methodology than that.
24. So in discussing these tricks, I'm going to just show you two examples of previous papers that use tricks like this to good effect.
25. So the first one I'll mention is a data augmentation-based approach in this paper, which focused on flying a drone through the forest.
26. So the output action space is discrete.
27. It's just go left, go straight or turn right.
28. And here's a video from their work.
29. So they're going to fly these drones through hiking trails in Switzerland, this is from the University of Zurich.
30. And the idea is pretty straightforward.
31. So they have a cognate and the cognate looks at the image and it predicts one of three discrete labels, left, right and straight.
32. Now, these are examples from the training set.
33. So they're labelled.
34. And where do they get the labels?
35. Well, they get the labels by, of course, using lots of machine learning using lots of hiking trails, but the data collection procedure is actually very straightforward.
36. They didn't actually have humans fly the quadcopter.
37. What they did instead is they got a person to walk the hiking trails, and the person was, let me fast forward here, wearing a funny hat.
38. Their hat had three cameras on it, a forward-facing camera, a left-facing camera, and a right-facing camera, and their approach was actually even simpler than the driving example.
39. They simply assumed that the person would always go in the correct direction, and they labeled the left-facing camera with the action to go right, and the right-facing camera with the action to go left, and the straight-facing camera with the action to go straight.
40. That's it.
41. That is the entirety of the method.
42. So there's no attempt to record the human's actions, and that actually worked pretty well, and I think this is a really nice illustration of how that data augmentation approach can enable imitation learning to work well, and it wouldn't surprise me in the least if, had they actually flown the quadcopter through the forest and only used a forward-facing camera, if their results would have actually been somewhat worse.
43. So this is a similar thing with a handheld camera.
44. Here is their drone.
45. Here's another interesting example.
46. This is a robotic manipulation example, and here the authors of this paper are using a very low cost, very cheap, and relatively inaccurate arm, and a very simplistic teleoperation system based on a kind of a hand motion detector, and they're teaching the robot various skills like using a cloth to wipe down a box of screwdrivers, picking up and pushing objects, things like this, and they're using a game controller here, and one of the things that they do is they illustrate a lot of mistakes in their demonstrations, kind of inevitably just because they have such a low cost and imperfect teleoperation system, and because they illustrate so many mistakes, they actually end up in a situation where the robot, when it makes mistakes, actually recovers from them.
47. So they have some examples where it picks up objects, sometimes it picks them up incorrectly, sometimes a human actually perturbs it, but the robot is actually pretty good at recovering from perturbations, including ones that are introduced by the person.
48. So here the person is messing with it, but the robot just is unworried about that and keeps trying to do the task.
49. So here it has to slide the wrench into a particular spot.
50. So sometimes imperfect data collection can actually be better than highly perfect data collection.
51. Okay, now that trick for getting behavioral cloning to work is not very reliable, and it takes a little bit of domain-specific expertise, although it does provide kind of a guidance anytime you're collecting data for imitation learning.
52. Keep in mind that having ways to put the system in states where the expert can demonstrate corrections can be a very good thing, and it's also worth thinking about data augmentation tricks.
53. But let's talk about some more technical solutions.
54. Why might you fail to fit the expert behavior?
55. Because if you can minimize the number, that value epsilon, perhaps even epsilon t squared might still be a small number.
56. So if you can understand why you might fail to fit the expert behavior, maybe you can get a model that's so powerful that it's probably that if mistakes are so low that even that quadratic cost doesn't actually worry you too much.
57. So why might you fail to fit the expert?
58. Well one reason is what I'll refer to as non-Markovian behavior.
59. Non-Markovian behavior means that the expert doesn't necessarily choose the action based only on the current state.
60. A second reason is multimodal behavior.
61. That means that the expert takes actions randomly and their distribution over actions is very complex and might have multiple modes.
62. Okay, let's talk about the non-Markovian behavior first.
63. So when we train a policy that is conditional on the current observation, the policy is Markovian in the sense that it assumes that the current observation is the only thing that matters.
64. It's basically assuming that the observation is the state.
65. That's not necessarily a problem if the expert also chose the action based only on the current observation.
66. But humans very rarely do that.
67. Humans can't really make decisions entirely in the moment, completely forgetting everything they saw before.
68. So if we see the same thing twice, if we were perfectly Markovian agents, we would do the same thing twice regardless of what happened before.
69. And that's pretty unnatural.
70. Oftentimes humans will base their decision on all of the past things they saw.
71. For example, if the human driver notices something in their blind spot and then looks back on the road, they still remember what they saw in their blind spot.
72. Or maybe even more problematic, if someone just cut them off and they got a little flustered, maybe they'll be driving a little differently for the next few seconds.
73. So generally humans are actually very non-Markovian in the sense that human behavior is very strongly affected by the temporal context.
74. So if we're training a policy that only looks at the current image, it's unaware of all that context.
75. And it might simply not be able to learn a single distribution that captures human behavior accurately because human behavior doesn't depend on just the current observation.
76. So how can we use the whole history?
77. Well, it's actually pretty straightforward.
78. We just need a policy representation that can read in the history of observations.
79. So we might have a variable number of frames.
80. And if we just simply combine all the frames into one giant image with like three thousand channels, that might be difficult because you might have too many weights.
81. So what we would typically do is use some kind of sequence model.
82. So we would have our, let's say if we're using images, we would have our convolutional encoder.
83. If you're not using images, you would have some other kind of encoder.
84. And you would encode all the past frames and then feed them through a sequence model such as an LSTM or a transformer, and then just predict the current action based on the entire sequence.
85. Setting up these models is a little bit involved, but there's actually nothing here that is special for imitation learning.
86. So the same way that you might build a sequence model to process, let's say, videos in supervised learning.
87. Exactly the same kinds of approaches can be used here, whether it's LSTMs or transformers or something else entirely like temporal convolutions.
88. Again, I won't talk about those architectural details in detail because they're actually not imitation learning specific.
89. So anything that you learned about before for sequence modeling could just as well be used for imitation learning.
90. There is however an important caveat that I want to mention, which is that using histories of observations does not always make things better and the reason that if sometimes makes things worse is that it might exacerbate correlations that occur in your data.
91. This is a little bit of an aside.
92. I don't necessarily expect all of you to kind of to know this in detail, but I do think it's an interesting aside.
93. It's something that perhaps might inspire some ideas for things like final projects.
94. Why might this work poorly?
95. Well, here's a little scenario.
96. Let's say that you have a strange kind of car where there's a dashboard indicator for whether you're pressing the brakes.
97. So whenever you press the brakes, there's a light bulb that lights up inside the cabin.
98. And the camera that is recording your data for imitation learning is inside the cabin so the camera can see out of the window and it can also see the brake indicator.
99. So whenever the person steps on the brake, the light lights up.
100. Now in this case there's a person standing in front of the car and the driver stepped on the brakes because there was a person there @ what the policy @ in the training data is it'll see one frame where the person is visible, but the brake indicator is not on, and the brake is pressed.
101. And then we'll see many steps after that where the brake indicator is pressed, is lit, and the brake is pressed.
102. So there's a very strong association between the brake indicator and the brake being pressed.
103. If you're reading in histories, the situation is a lot worse because now you don't even need the brake indicator.
104. When you're reading in histories, just the fact that the brake was pressed in previous time steps is apparent from looking at the sequential images.
105. You see the car slowing down, you know the brake was pressed.
106. So the point is that the action itself correlates with future instantiations of that action.
107. If the information is somehow hidden, then of course the policy is forced to pay attention to the important cue, which is the fact that there's a person.
108. But if these auxiliary cues are present, even though they are not the real cues that led to the action, they serve to confuse the policy as a spurious correlation, as a kind of causal confounder.
109. So the slowing down that you see when you look at history is caused by braking, but the policy might not realize that.
110. It might think that whenever you see the car slowing down, that's an indication that you should brake.
111. In the same way as the brake indicator is the effect, not the cause of braking, but when you see lots of images with that correlation in there, you might get confused.
112. So you can call this causal confusion as discussed a little bit in this paper.
113. There are a few questions we could ask about this.
114. Does including history mitigate causal confusion, or does it make it worse?
115. And I'll leave this as an exercise for you at home.
116. Another exercise is at the end of this lecture, we'll talk about a method called Dagger.
117. And after we talk about that, I want you to come back to this point and think about whether the method Dagger will actually address this problem or make it worse.
118. So I'll leave that as an exercise for you to think about.
119. All right, so that's non-Markovian behavior.
120. You can address it by using histories.
121. Keep in mind that that's not unequivocally always a good thing, but if you're what you're worried about is non-Markovian behavior, that's the thing to do.
122. Now let's talk about the other one, multimodal behavior.
123. That's kind of a subtle one.
124. Let's say that you want to navigate around a tree, maybe you're flying a quadcopter.
125. You can fly around the tree to the left, or you can fly around the tree to the right.
126. Both are valid solutions.
127. The trouble is that at that point when you're in front of the tree, some expert trajectories might involve going left, and some might involve going right.
128. So in general, in your training data, you'll see very different actions for very similar states.
129. Now this is not a problem if you're doing something like that Zurich paper, where they use a discrete action space, left, right, and straight, because you can easily represent a distribution where there's high probability for left, high probability for right, and low probability for straight, because you're directly outputting three numbers to indicate the probability of each of those actions.
130. However, if you're outputting a continuous action, maybe the mean and variance of a Gaussian distribution, now you have a problem, because a Gaussian has only one mode.
131. In fact, if you see examples of left and examples of right, and you average them together, that's very, very bad.
132. So how can we address this?
133. Well, we have a few choices.
134. We can use more expressive continuous distributions.
135. So instead of outputting the mean and variance of a single Gaussian, we can output something more elaborate.
136. Or we can actually use discretization, but make it feasible in high dimensional action spaces.
137. And I'll talk about both solutions a little bit next.
138. So first, let's talk about some examples of continuous distributions we can use.
139. And again, I won't go into great detail about each of these methods.
140. So for details about how to actually implement them, I'll have some pointers and references, and I would encourage you to look that up yourself if you want to actually try it.
141. My aim here is mostly to give you a survey level coverage of the different techniques so that you kind of know the right keywords and the right ideas.
142. Okay, so what we're going for is some way to set up a neural network.
143. So that it can output multiple nodes, for example, high probability of left, high probability of right, and low probability of going straight.
144. So we have a few options.
145. A very simple, but maybe less powerful option is to use a mixture of Gaussians.
146. And I'll talk about how to set that up with neural nets.
147. A more sophisticated one is to use latent variable models.
148. And then something that has recently become very popular is to use diffusion models because diffusion models have gotten a lot more...
149. effective and a lot easier to train in recent years.
150. But let's start with a mixture of Gaussians because that is probably the simplest thing to implement, although it's not quite as powerful as the others.
151. So the idea here is the following.
152. A mixture of Gaussians is... can be described as a set of means, covariances, and weights.
153. Okay?
154. So let's say you have n different Gaussians that you want to output.
155. Let's say, maybe it's n equals 10.
156. Let's say you're going to output 10 means, 10 covariances, and a weight on each of those 10 mixture elements to indicate how large each of them is.
157. So you probably learn about mixture of Gaussians in the context of something like clustering, where the means and the variances are just vectors and matrices that you learn.
158. Here, everything is conditional on the observation.
159. So your neural network is actually outputting the means and variances.
160. So they're not numbers that you store.
161. They're actually outputs of your neural net.
162. Before, our output was just the mean and maybe just one covariance matrix.
163. Now it's maybe going to be 10 vectors of means and 10 covariance matrices and a scalar weight on each of those 10 to indicate how large they are.
164. But in terms of implementing it, it's actually pretty straightforward.
165. All we have to do is code up our neural network so it has all those outputs, write down the equation for a mixture of Gaussians, take its logarithm as our training objective, and optimize it the same way that we did before.
166. So the way that you would implement this in, let's say, PyTorch is you would literally implement the equation for a mixture of Gaussians, take its log, and use that as your training objective.
167. Just don't forget to put a minus sign in front if you're minimizing it.
168. So that's basically the idea.
169. Your neural net outputs means, covariances, and weights.
170. And modern auto diff tools like PyTorch actually make this pretty easy to implement.
171. Of course, the problem with a mixture of Gaussians is that you choose a number of mixture elements, and that's how many modes you have.
172. So if you have 10 mixture elements and you want 10 modes, that's fine.
173. But what if you have 100 modes?
174. What if you have extremely high dimensional action spaces?
175. Perhaps you're not driving a car, but you're controlling a humanoid robot with 50 degrees of freedom, and you want 1,000 different modes.
176. Now you have to do something a little smarter.
177. Latent variable models provide us a way to represent a much broader class of distributions.
178. In fact, you can actually show that latent variable models can represent any distribution as long as the neural network is big enough.
179. The idea behind a latent variable model is that the output of the neural network is still a Gaussian, but in addition to the image, it receives another input, which is sampled from some prior distribution, like a zero mean unit variance Gaussian.
180. So you can think of it as almost like a random seed.
181. The random seed is passed into the network, and for different random seeds, it'll output different modes.
182. So for example, if we have a three-dimensional random vector that we put in, if we put in this random vector, we get this mode.
183. If we put in this random vector, then we get this other mode.
184. We have to train the network so that for different random vectors, it outputs different modes.
185. Now, unfortunately, you can't simply naively take the network and feed in random numbers and expect it to do this, because if you give it random numbers, those random numbers aren't actually correlated with anything in the input or output.
186. So if you just do this in the most obvious way, the neural net will actually ignore those numbers.
187. So the trick in training latent variable models is to make those numbers useful during training.
188. The most widely used type of method for this is what's called a conditional variational autoencoder.
189. We'll discuss conditional variational autoencoders in great detail much later on in the course in the second half.
190. So I won't actually describe how to make this work right now, but the high-level intuition is during training, the values of those random vectors that go into the network are not actually chosen randomly.
191. Instead, they're chosen in a smart way to correlate with which mode you're supposed to output.
192. So the idea is that during training, you figure out, this is particular training example as the left mode, this training example as the right mode, and you assign them different random vectors so that the neural net learns that it can pay attention to those random vectors, because they tell it which mode to output.
193. That's the intuition.
194. The particular technical way of making this work is a little bit more involved and requires more technical background, so we'll talk about that in the second half of the course.
195. But the high-level idea behind latent variable models is that you have an additional input to the model, and that additional input tells it which mode to output.
196. And then of course at test time, if you want to actually make this work, you can choose that random variable at random and then you'll randomly choose to go around the tree on the left or on the right.
197. The third class of distributions I'll talk about, which has gotten a lot of attention in recent years because these kinds of models have started working really well, is diffusion models.
198. Diffusion models are somewhat similar to latent variable models, but there are some differences.
199. Some of you might have heard about diffusion models as a way of generating images.
200. So things like DALL-E, stable diffusion, those are all methods that use diffusion models for image generation.
201. The way that diffusion models work for image generation, and this is a very high-level summary, so if this feels vague to you, it's because it is.
202. I could teach an entire class on diffusion models in principle, but for the purpose of covering it in two slides, I'm going to provide a very high-level overview.
203. Let's say that we have a particular training image.
204. I'm going to denote the training image as x0, and the subscript here doesn't denote time.
205. It denotes corruptions of the image.
206. So x0 is the least corrupted, xt is the most corrupted.
207. So x0 is the true image.
208. In a diffusion model, you construct a fake training set where you add noise to the image, and then you train the model to remove that noise.
209. So the image xi plus 1 is going to be the image xi plus noise.
210. So x1 is x0, which is the true image plus noise.
211. x2 is x1, which has some noise out of it, and it adds even more noise to it.
212. And if you take an image and you add these different amounts of noise, now you have a training set where you can teach a neural network to go backwards.
213. So the learned network looks at the image xi and it predicts xi minus 1.
214. So it's going to look at the slightly noisy image x1 and predicts x0.
215. It's going to look at the slightly more noisy image x2 and it's going to predict x1 and so on and so on.
216. In reality, we actually often train it to go all the way back to x0, so there's a choice to be made there, but for simplicity, it helps us to think about it as just going back one step.
217. And in reality, what we actually predict is just the noise itself.
218. And that's not actually that different, because if you predict xi minus 1, you can get that by just predicting the noise and just subtracting the noise from xi.
219. So you can either have f directly output xi minus 1 or you can have f output the noise, in which case xi minus 1 is just xi minus f .
220. And that's a much more common choice to make.
221. Okay, now this is image generation.
222. What we actually want to do is not generate images, we want to of course generate actions.
223. So what we can do is we can extend this framework to handle actions.
224. Now actions of course also have a temporal subscript, so I'm going to use the subscript t to denote time in the temporal process for control, and I'm going to use the second subscript to denote the diffusion time step.
225. So a t comma zero is the true action in the same way that x zero was the true image.
226. a t comma i plus one is a t i plus noise, and just like before we can learn a network that now takes in the current observation or state s t and a t i and it outputs a t i minus one.
227. Or just like before we would actually get it to output the noise, so then a t i minus one is a t i minus f of s t a t i.
228. So the training set is produced by taking all the actions, adding noise to them, teaching the network to predict what that noise was while also looking at the image, and then at test time if we want to actually figure out the action, then we feed in completely random noise and run this model for many steps to get it to denoise.
229. So turning the network over on its side, it gets as input a t i, it outputs a t i minus one, which is a slightly denoised version of that action, and it also gets to look at the image.
230. We start off with noise at test time, we feed that into this box as the first value of a t i, and then we repeat this process.
231. We denoise it, put it back in, denoise some more, and repeat this process many times, and then at the end out comes a clean, good action.
232. And during training, we add all this noise to ground truth actions, and we teach the network to undo the noise.
233. So that's the essential idea of a diffusion model.
234. Actually implementing it takes a number of additional design decisions, which I won't have time to go into here, but I'll reference some papers, and you can look at those papers for details.
235. The last trick I'm going to talk about is discretization.
236. Discretization is, in general, a very good way to get complex distributions, but it's very difficult to apply discretization naively in high dimensions.
237. So remember, in that Zurich paper, where the actions were to go left, go right, and go straight, this multimodality problem basically didn't exist.
238. But of course, that was for 1d actions.
239. In higher dimensions, if you have, let's say, 10-dimensional actions, discretizing the action space is impractical, because the number of bins you need increases exponentially, and so does the number of dimensions.
240. So the solution is to discretize one dimension at a time.
241. And that's the idea behind autoregressive discretization.
242. So here's how we can do it.
243. Let's say our action is a 3-dimensional vector.
244. I'm going to use at0 to denote dimension 0, at1 to denote dimension 1, and at2 to denote dimension 2.
245. Don't be confused with the notation from diffusion models before, this has nothing to do with that.
246. So the second number is just a dimension, and it's just a scalar value, right?
247. So here's how we're going to set up our network.
248. We take the image, and we encode it with some kind of encoder like a ConvNet, and then we put it into a sequence model, which could be a transformer or an LSTM, so whatever your favorite sequence model is.
249. And at the first step of the sequence, we output dimension 0, and we can discretize dimension 0 just into bins, right?
250. So it's just one dimension, discretizing a number line is pretty easy.
251. So we have one bin for every possible value.
252. You could have 10 bins, you could have 100 bins.
253. Since it's one-dimensional, it's very easy to do.
254. And then at the second time step in the sequence, we feed in the value at0, and we output at1, again with a discretization.
255. And at the next time step, we input at1, and we output at2.
256. So just like in a sequence model and something like a language model, you would output the next token, the next letter, here you would output the next dimension of the action space.
257. And now each dimension is discretized, and the number of bins is no longer exponential in the dimensionality, it's actually still linear in the dimensionality.
258. And then at test time, if we want a sample, then we do exactly what we do with any other sequence model, is instead of feeding in the ground truth value of each dimension, which we don't have, we would feed in the prediction from the previous time step.
259. Again, this is exactly the same as any other sequence model, like language models, for example.
260. So one way to implement is actually with a GPT-style decoder-only model.
261. Now, why does this work?
262. Well, the reason that this is a perfectly valid way to represent complex distributions is that it can be seen by looking at what probability is actually predicted each step.
263. So the first time step predicts p of at0 given st, because you get st as input, and your output is at0.
264. The second step predicts the probability of at1 given st and at0.
265. So the dependence on st comes from the fact that it's passed in through the sequence model, and at0 is fed as input.
266. The third time step, you predict at2 given st, at0, and at1.
267. If you multiply all these things together, then by the chain rule of probability, their product is exactly the probability of at0, at1, and at2 given st, which is exactly the probability of the action given the state.
268. So this is the policy.
269. If you multiply together, the probability is at all the time steps.
270. And that means that if you sample the different dimensions in sequence, then you will actually get a valid sample from the distribution p of at given st.
271. So autoregressive discretization is actually a great way to get complex distributions, but it's a little bit more heavyweight in the sense that you have to use these sequence models for outputting actions.
272. All right.
273. Next, let me talk about a few case studies of papers that use these kinds of formulations.
274. The first case study I'll talk about is a paper from 2023 from Chi et al.
275. that uses the fusion models to represent policies for robots.
276. So it works more or less exactly as you would expect.
277. There are two variants.
278. There is a convnet-based variant and a transformer-based variant, and they differ just in how they read in the image.
279. But in both cases, they read in the image and they perform this denoising operation.
280. So this is actually visualizing the denoising.
281. And the denoising process yields a short trajectory for the end effect to take over the next few steps.
282. And then the robot follows that trajectory and performs the task.
283. And they can use this for imitation learning, for learning things like picking up cups, or, you know, putting some sauce on a pizza.
284. This is maybe with like a cooperative person that kind of helps you make sure the sauce doesn't go all over the place.
285. But the point is that by using these fairly expressive multimodal diffusion policies, they can actually get pretty good behaviors out of their imitation learning system.
286. Here's an example of a method that uses latent variables.
287. Here, the latent variables are actually used in conjunction with the transformer, but it's not using that action discretization from before.
288. It's just outputting the continuous values of the actions.
289. So the transformer is just used to provide a more elaborate representation.
290. The latent variable here is this letter Z that you can see at the end of the input sequence.
291. And the paper treats it as a kind of style variable to account for the fact that human experts might perform the task with different styles on different trials.
292. And that's used to improve the expressivity of the model.
293. And imitation learning here uses this bimanual manipulation rig, which can then be used to provide demonstrations to teach the robot to do things like put a shoe on a foot.
294. So that's not a real foot, it's a mannequin foot.
295. And here the policy is inferring the latent variable just by sampling randomly from the prior test time.
296. But during training, it's using essentially a conditional VAE method of the sort that we'll describe later in the course.
297. So here's the shoe.
298. And it's, of course, going to buckle the shoe, because you need to make sure the shoe is buckled.
299. And here's another example, putting some batteries into a remote while an annoying graduate student distracts the robot by throwing objects into the background.
300. I guess this is the kind of thing you want to do to stress test your policies to make sure that they're pretty good with the distractors.
301. And here's the last case study that I'll present.
302. This is a autoregressive discretization method.
303. And in this work called RT1, the model is actually a transformer model that reads in a history of prior images and the language instruction and actually outputs a per-dimension discretization of the arm and base motion commands.
304. So it's a wheeled robot, so it can move the base and it can actually move the arm.
305. And all of those dimensions are discretized.
306. And that makes the entire control problem kind of a sequence to sequence problem.
307. So it's a sequence of images and text, converted into a sequence of per-dimension actions.
308. And because it's language condition, it can actually learn to perform a wide variety of different tasks when provided with a suitably large data set.
309. And when it's language condition, you can actually do fun things like connect it up to a large language model that will then parse complex instructions like bring me the rice chips from the drawer.
310. And then say, well, to do that, you have to first go to the drawer and then things like open the drawer.
311. And that's then commanded to this RT1 model, which then selects the per-dimension actions using that action discretization trick.
312. There's, of course, a lot more going on in this paper than just autoregressive discretization.
313. But I wanted to show this as just one example of that method really working in practice.
314. Okay, so I'll pause there and I'll resume in the next part.