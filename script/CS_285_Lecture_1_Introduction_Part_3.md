1. Why should we study deep reinforcement today?
2. Well, as I mentioned earlier, recent progress on data-driven large-scale AI systems has led to some pretty impressive results, but the methods that are trained to simply copy data produced by humans, they're mainly impressive because they produce things that look like human-generated results.
3. But in many cases we actually want algorithms that will do better than the typical human data, either because the human data is not good, or because it's hard to obtain, or because we really do want the highest possible performance, like in the case of AlphaGo.
4. We want solutions that are impressive because the machine didn't need to be told how to do something, because it discovered it on its own, because it discovered a solution that was better, or because it discovered a solution in a situation where it didn't have the benefit of human foresight to provide the kind of training data that it needed.
5. So recall that a lot of these very successful data-driven methods work on the basis of density estimation.
6. Which has particular implications.
7. It means that these methods will produce the kinds of data that humans tend to produce, but it also means that they in some sense won't go beyond good human behavior.
8. They might be much better at indexing into human data, as is certainly the case with large language models, they have a lot more knowledge, but not necessarily better at utilizing that knowledge to solve concrete problems.
9. If you tell a large language model, for example, to persuade somebody that it's, you know, in their best interest to go see a doctor, the language model probably won't be able to persuade them much better than a person would, despite the fact that it has this huge repository of internet knowledge to draw on.
10. So where does that leave us?
11. Well, we've got these data-driven AI systems that learn about the real world from data, potentially huge amounts of data, but they don't really try to be better than the data in any meaningful sense.
12. And we've got these reinforcement learning systems, and they can optimize a goal with emerging behavior, and that seems like something that should address one of the major shortcomings of these data-driven AI methods.
16. But of course, we need to figure out how to use these reinforcement learning methods at scale.
17. We need to combine them with the kinds of huge models and huge data sets that have been so successful.
18. And that's really where the deep part in deep reinforcement learning comes in.
19. So data-driven AI is all about using data, reinforcement learning is all about optimization.
20. Deep reinforcement learning is about this kind of optimization at scale.
21. And data without optimization basically doesn't allow us to solve new problems in new ways.
22. It might allow us to be very good at indexing into large data sets to figure out solutions that are human-like, but not necessarily solutions that are superhuman.
23. Something that I often like to bring up in the context of this discussion is an article written by Richard Sutton.
24. So Richard Sutton is actually one of the pioneers of reinforcement learning.
25. He was basically the person who popularized reinforcement learning in computer science, whereas previous to that it was really a subject of study primarily in psychology.
26. So in many ways he sort of founded the study of reinforcement learning in CS.
27. Richard Sutton wrote an article in 2019 called The Bitter Lesson.
28. Those of you that haven't read it, I very strongly encourage you to read through it.
29. It provides a very concise and very clear explanation for why we've seen this revolution in data-driven AI over the last few years.
30. And in that essay, he writes that we have to learn the bitter lesson, that building in how we think we think does not work in the long run.
31. The two methods that seem to scale arbitrarily are learning and search.
32. What he's arguing here is essentially that if we want very powerful learning machines, we should build machines that are very good at using data and very good at being scaled up, and not necessarily worry so much about engineering these systems so that they solve problems the way that we think that humans solve problems.
33. As an example, we might imagine building a system for detecting cars by somehow engineering some detectors for like wheels and headlights and things like that, and then try to program in that, well, a car is something that has four wheels and like two headlights in the front and two in the back.
34. So if you see some wheels and some headlights, well, that's probably a car.
35. And we can basically program that in, and that's actually how people used to build computer vision systems maybe about a decade ago.
36. But these days, we very rarely build perception systems that way.
38. Instead, what we do is we get lots of examples of cars, label them as cars, and let the computer figure it out.
39. And that's basically what Richard Sutton is saying, that let's not worry so much about building in how we think the problem should be solved, and let's instead focus on scalable learning machines.
40. The machine learning community has had sort of a perpetual debate about the degree to which we should be building in these kinds of components, and that's why this article was so influential.
41. But a lot of people who read this article take away kind of a funny impression, maybe that the emphasis is really on just scale and not really on the particular algorithm that is being scaled up.
42. So maybe it's okay if we just take, let's say, supervised learning methods, and as long as we can figure out how to basically shovel more data into GPUs or build larger server farms, that's really all that matters.
43. Data plus lots of machines, lots of computers, and not worry about how the problem is solved.
44. But that's not actually what the essay says.
45. Notice how it says learning and search.
46. It doesn't say learning and GPUs.
47. It doesn't say learning and big data sets.
48. It says learning and search.
49. And there's a very important reason for that.
50. Learning is about extracting patterns from data.
51. You look at the world, you pull in some data, and you train some learning machine on that, and it finds the patterns that are in there.
52. Search is about using computation to extract inferences.
53. Richard Sutton is using the term search in a very...
54. particular, very technical sense that is commonly used in reinforcement learning.
55. Search doesn't mean like A-star search necessarily.
56. Search means some kind of computation or optimization that you use to extract inferences.
57. So search is not about getting more data.
58. Search is about using what you've got to reach more interesting and more meaningful conclusions.
59. Search is essentially optimization.
60. Some kind of optimization process.
61. That uses typically iterative computation to make rational decisions.
62. And it's important to have both of those things.
63. Because learning is what allows you to understand the world.
64. And search is what allows you to leverage that understanding for interesting, emergent behavior.
65. And you really need both if you want to have flexible and rational and optimal decision making in real world settings.
66. You need to understand how the world works.
67. And then instead of just using your understanding to regurgitate what you've seen before, use that understanding to find a better solution.
68. Than what you've seen before.
69. That's basically what deep reinforcement learning tries to do.
70. Data without optimization doesn't allow us to solve new problems in new ways.
71. Optimization without data, without experience, is hard to apply in the real world.
72. Outside of things like simulators where you can write down equations of motion.
73. But if you have both of those things, then you can start to solve real world problems in more optimal ways.
74. I should add a little bit of philosophy here where this view is not just about how to control robots or how to play video games.
75. I specifically emphasized in the previous section that deep reinforcement learning methods have been applied very fruitfully to a range of other domains too.
76. And there's actually a deep reason for this.
77. To try to understand this reason, let's ask a very basic question.
78. Let's ask the question, why do we need machine learning?
79. And as an aside to help us answer that question, we can ask an even more basic question, why do we need brains?
80. The neuroscientist Daniel Walpert, who knows quite a bit about brains, had this to say on this topic.
81. He said, We have a brain for one reason and one reason that's to produce adaptable and complex movements.
83. Movement is the only way we have affecting the world around us.
84. And I believe that to understand movement is to understand the whole brain.
85. Now it won't surprise you to know that Daniel Walpert works on the neuroscience of motor control.
86. But I think this quote is very thought provoking.
87. And I think we can apply the same intuition to machine learning and formulate this postulate.
88. Perhaps we need machine learning for one reason and one reason only.
89. And that's to produce adaptable and complex decisions.
90. That makes a lot of sense.
91. In the same way that your brain is only useful to you insofar as it moves your body, because that's the only way that it affects the world around it.
92. The machine learning system is only useful insofar as it makes good decisions, because that's the only thing it's outputting.
93. And now we can start to view all machine learning problems through this lens, not as problems of prediction, but as problems of decision making.
94. This is obvious if you're controlling a robot, your decision is how to move the joints.
95. It's obvious if you're driving a car, your decision is how to steer the car.
96. But even something like a computer vision system, in the end is a decision making system.
97. It may make a decision, which could be the image label, but really the decision has implications of what happens downstream of that image label.
98. Maybe this perception system is detecting how many cars there are at an intersection, and that label will be used to determine how to route traffic.
99. So it has long term implications.
100. Maybe the computer vision system is detecting people in a security camera, and it's going to call security if it sees someone where they shouldn't be.
101. Well, that's definitely a decision that could lead to some very complex and very difficult to model outcomes.
102. If you view all of the outcomes of machine learning problems as decisions, then it becomes clear that all machine learning problems are really reinforcement learning problems in disguise.
103. It's just that in some cases we have the privilege of supervised labeled data that can aid us in solving them.
104. And while this perspective might be a little bit reductionist, I think it's important to keep in mind because it really tells us those building blocks, learning and search, are not just special things that we want for robots and video game playing, but they're really general building blocks of AI systems.
105. And that brings us to some big questions like how do we build intelligent machines?
106. Very general intelligent machines, not just machines that can detect objects and images, but things like this, or this, or this, or if you are more nefariously inclined, things like this.
107. The kinds of intelligent machines that were popularized in science fiction that captured the imagination, maybe they're quite a ways away, but how do we start taking steps towards this kind of thing?
113. I think deep reinforcement learning forms a significant part of that.
114. And I think if we study it now, we might put ourselves on the path to eventually answer some pretty fundamental questions.
115. So, why should we study deep reinforcement learning today?
116. Well, part of the answer is that big end-to-end trained models seem to work quite well.
117. If we use large data sets and large models like transformers, we can solve some pretty impressive problems.
118. And at the same time, we have RL algorithms that we can feasibly combine with deep neural networks.
119. We've figured out a lot about how to implement RL algorithms so they can be used to train these kind of big end-to-end models.
120. And yet, learning-based control in truly open world settings remains a major open challenge.
121. There are some initial results, including the robotics results I presented, the results in other domains, that show the inkling of the capability of these systems.
122. But a lot of potential has yet to be realized.
123. And I'll talk about some of that potential in today's lecture and also over the course of this class.
124. And also discuss how some of these ideas can maybe bring us closer.
125. So it's a very exciting time, I think, to study this topic because in some ways many of the puzzle pieces are falling into place.
126. And yet, major questions remain, which could be questions that you yourselves could answer in your own future work.
127. But before I get into that, I want to discuss a little bit about the broader picture of the reinforcement learning field.
128. Besides the basic problem of maximizing reward functions, what are other problems that we need to solve to enable real-world sequential decision making?
129. Because this question is not just about reward maximization.
131. It's also about a variety of other problems that crop up when we study decision making and control in realistic data-driven settings and the kinds of methods that could address it.
132. For example, basic reinforcement learning deals with maximizing rewards.
133. But this is not the only problem that matters for sequential decision making.
134. We'll cover more advanced topics like learning reward functions from examples, which is referred to as inverse reinforcement learning.
135. Transferring knowledge between domains, like transfer learning and meta-learning.
136. Learning to predict and using prediction to act.
137. And so on.
138. Here's one question.
139. Where do rewards come from?
140. If you're playing a video game, it's pretty obvious.
141. Maybe the reward function is the score in the video game.
142. You kind of don't have to think about it very hard.
143. But in other settings, you do.
144. What if you want to get a robot to pick up a jug and pour a glass of water?
145. Well, any child could do this.
146. But just figuring out the reward function, is the water in the glass, is itself a complex perception problem.
147. There's a paper that was published by some folks at UC Berkeley on exploration, actually, about four or five years ago.
148. And it had this nice quote.
149. As human agents, we are accustomed to operating with rewards that are so sparse that we only experience them once or twice in a lifetime, if at all.
150. What this means is that a lot of the things that humans do that are very impressive, their reward might be so delayed that it's very difficult to imagine learning just from that reward signal.
151. For example, the reward that you'll receive for, let's say, completing a PhD degree.
153. You only get that reward once, and you maybe experience some satisfaction.
154. The real outcome might be what you do afterwards with that degree.
155. And yet, you might set yourself on the path to do that.
156. Clearly, it's not something that you learn through trial and error by attempting many, many PhD degrees in the past.
157. This is actually a quote that was posted on Reddit, where the commenter replied by writing, I pity the author.
158. So we know that there is actually a structure in the human brain, the basal ganglia, which is actually responsible for the reward signal that the brain uses for reinforcement learning.
160. This is actually something that's been studied quite a lot.
161. And it's a non-trivial structure.
162. You can see it takes up quite a bit of space.
163. So clearly, it's doing something sophisticated.
164. And it's not hard to imagine that, for example, for a cheetah that needs to chase down a gazelle, well, if the cheetah learned through trial and error, receiving the reward only when it caught the gazelle, that's a pretty ridiculous image of a learning system.
165. If the cheetah just runs around in the savannah randomly, hoping to randomly stumble into a gazelle, then randomly eat it, and only then realize that catching gazelles is a good idea, well, that cheetah would probably die of starvation.
166. Of course cheetahs don't learn in this way.
167. They might learn from observing other cheetahs.
168. They might learn from their parents.
169. They might learn from all sorts of other signals.
170. But clearly, they're not learning from rewards obtained only from eating the meat of the gazelle at the end of a successful hunt.
171. So there's a lot that goes into these reward signals.
172. And then there's the other thing.
173. And you could imagine extracting other, more useful forms of supervision.
174. You could learn from demonstrations, either by directly copying the observed behavior or even inferring rewards from observed behavior by something called inverse reinforcement learning.
175. You could learn from observing the world, learn to predict what will happen next, even if you're not sure what you're supposed to be doing, and then leverage that knowledge later, once you're more aware of what your task is.
176. You can employ unsupervised learning, unsupervised feature extraction, things like that.
177. You can also transfer knowledge from other tasks.
178. You can even use meta-learning, where you learn to adapt more quickly from your past experience of solving other tasks.
179. And these are all things that we could try to leverage, and these are all things that we'll actually learn about in this course.
180. Here's an example of imitation learning.
181. This is actually a fairly old example, at this point from about 80 years ago, from some work from NVIDIA, showing a purely imitation-based method for autonomous driving.
182. Now, this method tries to directly copy the actions of the observed human driver.
183. But of course, you could do a lot better.
184. You could, for example, infer their intent.
186. This is a psychology study.
187. Here, the test subject is the child on the right-hand side.
188. Now, you can see the child here is not going to try to imitate what the experimenter is doing, because clearly the experimenter is not doing something very smart.
189. What the child will do instead is infer their intent, and then taking a very different sequence of actions that is better for fulfilling their intent, rather than simply copying them.
190. This is really the hallmark of human imitation.
191. When we say that a person imitates somebody else, they're not literally observing someone's muscle activations and performing the same muscle activations.
193. At some level, they're always inferring something about what that other creature or person is attempting to do, and then doing it in their own way.
194. It might be very literal, where they still carry out the same motions, but figure out the commands to their muscles that will create those motions.
195. Or it might be even more abstract, like it is here, where they carry out entirely different actions, but that lead to the seemingly desired outcome.
196. Inverse reinforcement learning algorithms can be actually used with robots.
197. This is, again, work that's at this point pretty old.
199. It's about eight years old.
200. It shows an inverse reinforcement learning algorithm where this robot infers the intent of the human demonstrator, showing this pouring motion, and figures out that the point is to really seek out that yellow cup, and to pour the content of the orange cup into the yellow cup.
201. Once it inferred that intent, then it could perform the task in a variety of settings.
202. Prediction is a really big part of control.
203. Prediction is separate from how we typically think of model-free reinforcement learning.
204. But there's ample evidence in neuroscience and psychology that prediction is a very important part of how humans and animals learn about their world.
206. We could imagine predictive models in a very literal sense, where you could actually predict your future sensory readings.
207. And you can implement real-world predictive models.
208. So here, a robot plays around with objects in its environment, collects some data, and then learns to predict what it will see in response to different actions.
209. So the different columns here show predicted future images in response to different motor commands.
210. This is quite a while back this is seven years ago.
210. So you can see that the predictions here are not of very high quality, but they capture the gist of what the robot is trying to do.
211. And they can be used to control objects.
212. So you can tell it, move this particular object marked in red to the green location.
213. It'll imagine the movement, and then it will actually actuate the arm to move the object in that way.
214. So predictive models can allow you to solve new tasks.
215. You can use this as a very powerful tool for emergent behavior.
218. You could, for example, command the robot to move some objects, and it might figure out that it needs to pick up a tool to move those two objects together.
219. Here's another tool use example here.
220. It figures out that that L-shaped tool can slide the blue object.
221. And here, there's an emergent tool use scenario where it figures out that to move these two pieces of trash, the water bottle makes for a nice improvised tool.
222. And predictive models have really come a long way.
223. So in recent years, we've been able to do a lot better with modern advances in general modeling.
224. This is a diffusion-based video prediction model that is being used to synthesize clips of driving videos.
225. The first three frames here are real.
226. The remaining frames are actually synthesized.
227. And you can see that the model will actually produce realistic camera movement.
228. It will introduce new objects as the car turns.
229. It will even predict the motion of the other cars with some reasonable fidelity.
230. In these examples, by the way, the left video in each pair is the real one.
231. The right one is the synthetic one.
232. And here, the same model is being run on robotic videos similar to the ones that I showed before, just so you can see the contrast from 2017 to 2022.
233. You can see that now the arm is clear and crisp.
234. The objects move in realistic ways and so forth.
235. There's also a lot of interesting progress, especially in the past year, on leveraging advances in pre-trained models.
236. So when we do reinforcement learning, we typically don't have to do it from scratch.
237. What we could do is we could use a model pre-trained on large amounts of Internet data and then use it for control.
238. This is actually an imitation learning example.
239. It doesn't do RL.
240. It actually does direct imitation, but it is doing a learning-based control.
241. This is the RT2 model, which uses a first a language model that is pre-trained on language, then a visual language model that uses that language model to process Internet images for things like question and answering, like what is happening in the image.
242. Let's say it's a great doggy walking down the street.
243. So now the model understands pictures.
244. It understands text.
245. And then that model is further fine-tuned to output robot actions so that when it's told what should the robot do to pick up the chips, it'll output the numerical values for the actions that will actually pick up the chips.
246. So now it can bring in knowledge that it learned from the Internet to perform this task more effectively.
247. Here are some examples of the kind of intelligence tests that this model can pass.
248. So it can be told to move the banana to the bottle.
249. The robot data has examples of moving bananas, but to understand what it means to move it to the bottle, it has to leverage Internet data.
250. Here it's asked to solve a math problem by putting the banana on the answer to the math problem.
251. Here it's told to put the strawberry into the correct bowl.
252. To figure out what correct bowl means, it needs to recognize the fruits in each of the bowls and figure out that the strawberry bowl is in fact the correct one.
253. And here are some more examples.
254. Pick up an object that is different from all the other objects.
255. Now it knows how to pick up objects in the robot data, but it doesn't know what different from all the others means from that.
256. But that has to leverage Internet data, and it figures out that the bar is the different object because all the other objects are bottles.
257. It can understand instruction in other languages, even though the robot data is only annotated in one language and so on.
258. Okay, so these are some examples of the kinds of problems that we might study in the context of learning-based decision making, besides the 4RL problems.
259. But to conclude this lecture, I want to end on maybe a somewhat more grandiose point.
260. I want to come back to this question, how do we build intelligent machines, and really argue that the basic building blocks of DeepRL might be very good building blocks for answering this question.
261. This is of course a controversial statement, I don't expect everybody to agree with me on this, but this is a big part of why I'm excited about this topic, and I hope to convey some of that excitement to you.
262. So, imagine that you have to build an intelligent machine, something as intelligent as a person.
263. Where would you start?
264. Well, in the olden days, the way we would think about this is that maybe we need to understand the brain, and the brain has a lot of parts, so let's understand what those parts are, figure out how each of them work, and then write computer programs to emulate the behavior of each of those parts.
265. Of course, our modern understanding of the brain is more advanced than what it was in the 19th century, but parts of the brain more closely reflect their actual function.
266. But this is still a very difficult problem, because each of the parts is very complex, and if we have to do a bunch of programming to code up the behavior of each of the parts, and do a bunch more coding to wire them together, we might be at this for a very long time.
267. That might just be a very, very difficult way to implement an intelligent machine.
268. It might actually take a lot more intelligence on our part than we actually have.
269. So, if we hypothesize that learning might be the basis of intelligence, that might actually offer us a much easier way to address this problem.
270. And here's an argument for why learning might be the basis of intelligence.
271. There are some things that we can all do, like walking.
272. So, it might be reasonably argued that maybe those things are sort of built into our brains somehow.
273. But there are also some things that we can only learn, like driving a car.
274. Clearly, driving a car is not built into our brains, because cars weren't around when our brains evolved.
275. And we can learn a huge variety of things, including very difficult things.
276. So, therefore, our learning mechanisms are likely powerful enough to do everything that we associate with intelligence.
277. It may be that in practice we don't actually use our learning mechanisms for some things, like walking, but we might hypothesize that maybe they're powerful enough that if we didn't have those things built in, we could figure it out anyway.
278. That may or may not be true, but I think there's a pretty good reason to believe this might be true.
279. It might still be very convenient to hard-code a few really important bits, but let's not get distracted by that part.
280. We can further hypothesize that not only is learning the basis of intelligence, but in fact maybe there's actually a single learning procedure that underlies all that we associate with intelligent behavior.
281. Now, that's a more radical statement.
282. It basically says that the way that we learn how to see and the way that we learn how to talk and the way that we learn how to hear is at some level the same.
283. Instead of having an algorithm for every module, maybe we have a single flexible algorithm that placed in the right context implements all of the modules, everything that we need in the brain.
284. And there's some circumstantial evidence to indicate that this might in fact be the case.
285. So, for example, these are some slides borrowed from Andrew Ng.
286. You can build an electrode array that you can put on your tongue, attach that array to a camera, and learn how to perceive visual percepts through your tongue.
287. You can take an animal, a ferret, you can disconnect the optic nerve from the visual cortex and plug it into the auditory cortex, and after a while the ferret will regain some degree of visual acuity, which means that its auditory cortex can essentially learn to process visual signals.
288. So these things kind of indicate that perhaps there's a degree of generality or homogeneity to the brain, for the neural cortex, such that it can adapt to whatever sensory input is provided, which might indicate that there's one algorithm.
289. And if there is one algorithm, what does this one algorithm need to be able to do?
290. Well, it needs to interpret rich sensory inputs, and it needs to choose complex actions.
291. And to do both of those things, we need large high-capacity models, because that's the only way we know how to deal with rich sensory inputs, and we need reinforcement learning, because that's the mathematical formulas we use to take actions.
292. So, why deep reinforcement learning?
293. Well, the deep part provides us with scalable learning from large complex data sets, and the reinforced learning gives us the optimization, the ability to take actions.
294. The combination of learning and search.
295. Deep is great for learning.
296. Reinforced learning is the way that we do the search.
297. And in fact, there is some evidence in neuroscience for both these things.
298. There's evidence that the kinds of representations acquired by deep neural networks have some statistical similarity to representations that are observed in the brain.
299. That doesn't mean that the brain works the same way that deep nets do, it just means that at some level, when you process lots of data and extract suitable representations, they end up looking similar.
300. Which could have more to do with the fact that a large enough learning machine just pulls out those patterns in the data, because that's what the data is made of.
301. Or it could say something about deep learning, that's, I think, a much harder question to answer, but the evidence suggests that some kind of representational similarity exists for visual percepts, for auditory features, and even for the sense of touch.
302. The experiments done to ascertain this are actually a little bit creative, where the brain signals indicating the kind of features that, in this case, monkeys use for touch, are obtained from recordings from monkey neurons.
303. The deep learning experiment is done by actually taking a glove dusted with white dust, getting a person to touch objects, and then using a deep neural network to discover patterns in the dust patterns on the glove.
304. So, interesting experiment, suggests that maybe the statistical properties of features extracted by sufficiently powerful learning machines resemble the features in the brain.
305. And there's plenty of evidence in favor of reinforcement learning as at least one of the mechanisms underlying decision making in humans and animals.
306. In fact, reinforcement learning actually emerged as a study of animal intelligence, but we know now from evidence that percepts that anticipate reward become associated with similar firing patterns as the reward itself, which is exactly what we would expect from a temporal difference learning process.
307. The basal ganglia appears to be a kind of reward system, and that model-free RL-like adaptation is often a good fit for experimental data of animal adaptation.
308. Although not always.
309. But the picture is not complete, right?
310. So all of these bits of circumstantial evidence might suggest that the tools of deep learning and reinforcement learning might be good tools for tackling the problem of intelligence.
311. But the problem is clearly not solved.
312. We have great methods that can learn from huge amounts of data by using deep learning, we have great optimization methods for RL, we don't yet have amazing methods that both use data in RL.
313. RL has been made much more scalable in recent years, it can tackle things like real-world robotics problems, but the kind of huge-scale language model and generative modeling applications still primarily use supervised learning, so there are still some algorithmic building blocks that are necessary.
314. And furthermore, humans learn incredibly quickly, whereas deep RL methods typically require large amounts of data.
315. And humans reuse past knowledge, whereas transfer learning in RL is still an open problem.
316. It's not always clear what the reward function should be, and it's not always clear what the role of prediction should be.
317. It seems like these methods can be very powerful, but how do they fit in with model-free methods?
318. Are they just different things, or can they be reconciled in some way?
319. So all of these question marks, I think, give us ample space for additional research that we can do in this area, and perhaps if the tools of deep learning and reinforcement learning are the right tools for building enormously powerful artificial intelligence systems, then maybe studying these questions can allow us to make some headway on that problem.
320. And ultimately, I think that we can get away from this picture of thinking of intelligence systems as a collection of modules to implement, and instead as a very elegant and simple framework where we have a general learning algorithm that can figure out whatever problems is posed to it.
321. In fact, this idea is not by any means new.
322. It's not something that was created in the 21st century.
323. It's not even something that was created for deep learning or even in the age of machine learning.
324. Here's a quote that I think very nicely exemplifies this perspective.
325. Instead of trying to produce a program to simulate the adult mind, why not rather try to produce one which simulates the child's?
326. If this were then subjected to an appropriate course of education, one would obtain the adult brain.
327. Who said this?
328. Alan Turing.