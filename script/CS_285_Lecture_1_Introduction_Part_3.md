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
13. But of course, we need to figure out how to use these reinforcement learning methods at scale.
14. We need to combine them with the kinds of huge models and huge data sets that have been so successful.
15. And that's really where the deep part in deep reinforcement learning comes in.
16. So data-driven AI is all about using data, reinforcement learning is all about optimization.
17. Deep reinforcement learning is about this kind of optimization at scale.
18. And data without optimization basically doesn't allow us to solve new problems in new ways.
19. It might allow us to be very good at indexing into large data sets to figure out solutions that are human-like, but not necessarily solutions that are superhuman.
20. Something that I often like to bring up in the context of this discussion is an article written by Richard Sutton.
21. So Richard Sutton is actually one of the pioneers of reinforcement learning.
22. He was basically the person who popularized reinforcement learning in computer science, whereas previous to that it was really a subject of study primarily in psychology.
23. So in many ways he sort of founded the study of reinforcement learning in CS.
24. Richard Sutton wrote an article in 2019 called The Bitter Lesson.
25. Those of you that haven't read it, I very strongly encourage you to read through it.
26. It provides a very concise and very clear explanation for why we've seen this revolution in data-driven AI over the last few years.
27. And in that essay, he writes that we have to learn the bitter lesson, that building in how we think we think does not work in the long run.
28. The two methods that seem to scale arbitrarily are learning and search.
29. What he's arguing here is essentially that if we want very powerful learning machines, we should build machines that are very good at using data and very good at being scaled up, and not necessarily worry so much about engineering these systems so that they solve problems the way that we think that humans solve problems.
30. As an example, we might imagine building a system for detecting cars by somehow engineering some detectors for like wheels and headlights and things like that, and then try to program in that, well, a car is something that has four wheels and like two headlights in the front and two in the back.
31. So if you see some wheels and some headlights, well, that's probably a car.
32. And we can basically program that in, and that's actually how people used to build computer vision systems maybe about a decade ago.
33. But these days, we very rarely build perception systems that way.
34. Instead, what we do is we get lots of examples of cars, label them as cars, and let the computer figure it out.
35. And that's basically what Richard Sutton is saying, that let's not worry so much about building in how we think the problem should be solved, and let's instead focus on scalable learning machines.
36. The machine learning community has had sort of a perpetual debate about the degree to which we should be building in these kinds of components, and that's why this article was so influential.
37. But a lot of people who read this article take away kind of a funny impression, maybe that the emphasis is really on just scale and not really on the particular algorithm that is being scaled up.
38. So maybe it's okay if we just take, let's say, supervised learning methods, and as long as we can figure out how to basically shovel more data into GPUs or build larger server farms, that's really all that matters.
39. Data plus lots of machines, lots of computers, and not worry about how the problem is solved.
40. But that's not actually what the essay says.
41. Notice how it says learning and search.
42. It doesn't say learning and GPUs.
43. It doesn't say learning and big data sets.
44. It says learning and search.
45. And there's a very important reason for that.
46. Learning is about extracting patterns from data.
47. You look at the world, you pull in some data, and you train some learning machine on that, and it finds the patterns that are in there.
48. Search is about using computation to extract inferences.
49. Richard Sutton is using the term search in a very...
50. particular, very technical sense that is commonly used in reinforcement learning.
51. Search doesn't mean like A-star search necessarily.
52. Search means some kind of computation or optimization that you use to extract inferences.
53. So search is not about getting more data.
54. Search is about using what you've got to reach more interesting and more meaningful conclusions.
55. Search is essentially optimization.
56. Some kind of optimization process.
57. That uses typically iterative computation to make rational decisions.
58. And it's important to have both of those things.
59. Because learning is what allows you to understand the world.
60. And search is what allows you to leverage that understanding for interesting, emergent behavior.
61. And you really need both if you want to have flexible and rational and optimal decision making in real world settings.
62. You need to understand how the world works.
63. And then instead of just using your understanding to regurgitate what you've seen before, use that understanding to find a better solution.
64. Than what you've seen before.
65. That's basically what deep reinforcement learning tries to do.
66. Data without optimization doesn't allow us to solve new problems in new ways.
67. Optimization without data, without experience, is hard to apply in the real world.
68. Outside of things like simulators where you can write down equations of motion.
69. But if you have both of those things, then you can start to solve real world problems in more optimal ways.
70. I should add a little bit of philosophy here where this view is not just about how to control robots or how to play video games.
71. I specifically emphasized in the previous section that deep reinforcement learning methods have been applied very fruitfully to a range of other domains too.
72. And there's actually a deep reason for this.
73. To try to understand this reason, let's ask a very basic question.
74. Let's ask the question, why do we need machine learning?
75. And as an aside to help us answer that question, we can ask an even more basic question, why do we need brains?
76. The neuroscientist Daniel Walpert, who knows quite a bit about brains, had this to say on this topic.
77. He said, We have a brain for one reason and one reason that's to produce adaptable and complex movements.
78. Movement is the only way we have affecting the world around us.
79. And I believe that to understand movement is to understand the whole brain.
80. Now it won't surprise you to know that Daniel Walpert works on the neuroscience of motor control.
81. But I think this quote is very thought provoking.
82. And I think we can apply the same intuition to machine learning and formulate this postulate.
83. Perhaps we need machine learning for one reason and one reason only.
84. And that's to produce adaptable and complex decisions.
85. That makes a lot of sense.
86. In the same way that your brain is only useful to you insofar as it moves your body, because that's the only way that it affects the world around it.
87. The machine learning system is only useful insofar as it makes good decisions, because that's the only thing it's outputting.
88. And now we can start to view all machine learning problems through this lens, not as problems of prediction, but as problems of decision making.
89. This is obvious if you're controlling a robot, your decision is how to move the joints.
90. It's obvious if you're driving a car, your decision is how to steer the car.
91. But even something like a computer vision system, in the end is a decision making system.
92. It may make a decision, which could be the image label, but really the decision has implications of what happens downstream of that image label.
93. Maybe this perception system is detecting how many cars there are at an intersection, and that label will be used to determine how to route traffic.
94. So it has long term implications.
95.  Maybe the computer vision system is detecting people in a security camera, and it's going to call security if it sees someone where they shouldn't be.
96.  Well, that's definitely a decision that could lead to some very complex and very difficult to model outcomes.
97.  If you view all of the outcomes of machine learning problems as decisions, then it becomes clear that all machine learning problems are really reinforcement learning problems in disguise.
98.  It's just that in some cases we have the privilege of supervised labeled data that can aid us in solving them.
99.  And while this perspective might be a little bit reductionist, I think it's important to keep in mind because it really tells us those building blocks, learning and search, are not just special things that we want for robots and video game playing, but they're really general building blocks of AI systems.
100. And that brings us to some big questions like how do we build intelligent machines?
101. Very general intelligent machines, not just machines that can detect objects and images, but things like this, or this, or this, or if you are more nefariously inclined, things like this.
102. The kinds of intelligent machines that were popularized in science fiction that captured the imagination, maybe they're quite a ways away, but how do we start taking steps towards this kind of thing?
103. I think deep reinforcement learning forms a significant part of that.
104. And I think if we study it now, we might put ourselves on the path to eventually answer some pretty fundamental questions.
105. So, why should we study deep reinforcement learning today?
106. Well, part of the answer is that big end-to-end trained models seem to work quite well.
107. If we use large data sets and large models like transformers, we can solve some pretty impressive problems.
108. And at the same time, we have RL algorithms that we can feasibly combine with deep neural networks.
109. We've figured out a lot about how to implement RL algorithms so they can be used to train these kind of big end-to-end models.
110. And yet, learning-based control in truly open world settings remains a major open challenge.
111. There are some initial results, including the robotics results I presented, the results in other domains, that show the inkling of the capability of these systems.
112. But a lot of potential has yet to be realized.
113. And I'll talk about some of that potential in today's lecture and also over the course of this class.
114. And also discuss how some of these ideas can maybe bring us closer.
115. So it's a very exciting time, I think, to study this topic because in some ways many of the puzzle pieces are falling into place.
116. And yet, major questions remain, which could be questions that you yourselves could answer in your own future work.
117. But before I get into that, I want to discuss a little bit about the broader picture of the reinforcement learning field.
118. Besides the basic problem of maximizing reward functions, what are other problems that we need to solve to enable real-world sequential decision making?
119. Because this question is not just about reward maximization.
120. It's also about a variety of other problems that crop up when we study decision making and control in realistic data-driven settings and the kinds of methods that could address it.
121. For example, basic reinforcement learning deals with maximizing rewards.
122. But this is not the only problem that matters for sequential decision making.
123. We'll cover more advanced topics like learning reward functions from examples, which is referred to as inverse reinforcement learning.
124. Transferring knowledge between domains, like transfer learning and meta-learning.
125. Learning to predict and using prediction to act.
126. And so on.
127. Here's one question.
128. Where do rewards come from?
129. If you're playing a video game, it's pretty obvious.
130. Maybe the reward function is the score in the video game.
131. You kind of don't have to think about it very hard.
132. But in other settings, you do.
133. What if you want to get a robot to pick up a jug and pour a glass of water?
134. Well, any child could do this.
135. But just figuring out the reward function, is the water in the glass, is itself a complex perception problem.
136. There's a paper that was published by some folks at UC Berkeley on exploration, actually, about four or five years ago.
137. And it had this nice quote.
138. As human agents, we are accustomed to operating with rewards that are so sparse that we only experience them once or twice in a lifetime, if at all.
139. What this means is that a lot of the things that humans do that are very impressive, their reward might be so delayed that it's very difficult to imagine learning just from that reward signal.
140. For example, the reward that you'll receive for, let's say, completing a PhD degree.
141. You only get that reward once, and you maybe experience some satisfaction.
142. The real outcome might be what you do afterwards with that degree.
143. And yet, you might set yourself on the path to do that.
144. Clearly, it's not something that you learn through trial and error by attempting many, many PhD degrees in the past.
145. This is actually a quote that was posted on Reddit, where the commenter replied by writing, I pity the author.
146. So we know that there is actually a structure in the human brain, the basal ganglia, which is actually responsible for the reward signal that the brain uses for reinforcement learning.
147. This is actually something that's been studied quite a lot.
148. And it's a non-trivial structure.
149. You can see it takes up quite a bit of space.
150. So clearly, it's doing something sophisticated.
151. And it's not hard to imagine that, for example, for a cheetah that needs to chase down a gazelle, well, if the cheetah learned through trial and error, receiving the reward only when it caught the gazelle, that's a pretty ridiculous image of a learning system.
152. If the cheetah just runs around in the savannah randomly, hoping to randomly stumble into a gazelle, then randomly eat it, and only then realize that catching gazelles is a good idea, well, that cheetah would probably die of starvation.
153. Of course cheetahs don't learn in this way.
154. They might learn from observing other cheetahs.
155. They might learn from their parents.
156. They might learn from all sorts of other signals.
157. But clearly, they're not learning from rewards obtained only from eating the meat of the gazelle at the end of a successful hunt.
158. So there's a lot that goes into these reward signals.
159. And then there's the other thing.
160. And you could imagine extracting other, more useful forms of supervision.
161. You could learn from demonstrations, either by directly copying the observed behavior or even inferring rewards from observed behavior by something called inverse reinforcement learning.
162. You could learn from observing the world, learn to predict what will happen next, even if you're not sure what you're supposed to be doing, and then leverage that knowledge later, once you're more aware of what your task is.
163. You can employ unsupervised learning, unsupervised feature extraction, things like that.
164. You can also transfer knowledge from other tasks.
165. You can even use meta-learning, where you learn to adapt more quickly from your past experience of solving other tasks.
166. And these are all things that we could try to leverage, and these are all things that we'll actually learn about in this course.
167. Here's an example of imitation learning.
168. This is actually a fairly old example, at this point from about 80 years ago, from some work from NVIDIA, showing a purely imitation-based method for autonomous driving.
169. Now, this method tries to directly copy the actions of the observed human driver.
170. But of course, you could do a lot better.
171. You could, for example, infer their intent.
172. This is a psychology study.
173. Here, the test subject is the child on the right-hand side.
174. Now, you can see the child here is not going to try to imitate what the experimenter is doing, because clearly the experimenter is not doing something very smart.
175. What the child will do instead is infer their intent, and then taking a very different sequence of actions that is better for fulfilling their intent, rather than simply copying them.
176. This is really the hallmark of human imitation.
177. When we say that a person imitates somebody else, they're not literally observing someone's muscle activations and performing the same muscle activations.
178. At some level, they're always inferring something about what that other creature or person is attempting to do, and then doing it in their own way.
179. It might be very literal, where they still carry out the same motions, but figure out the commands to their muscles that will create those motions.
180. Or it might be even more abstract, like it is here, where they carry out entirely different actions, but that lead to the seemingly desired outcome.
181. Inverse reinforcement learning algorithms can be actually used with robots.
182. This is, again, work that's at this point pretty old.
183. It's about eight years old.
184. It shows an inverse reinforcement learning algorithm where this robot infers the intent of the human demonstrator, showing this pouring motion, and figures out that the point is to really seek out that yellow cup, and to pour the content of the orange cup into the yellow cup.
185. Once it inferred that intent, then it could perform the task in a variety of settings.
186. Prediction is a really big part of control.
187. Prediction is separate from how we typically think of model-free reinforcement learning.
188. But there's ample evidence in neuroscience and psychology that prediction is a very important part of how humans and animals learn about their world.
189. We could imagine predictive models in a very literal sense, where you could actually predict your future sensory readings.
190. And you can implement real-world predictive models.
191. So here, a robot plays around with objects in its environment, collects some data, and then learns to predict what it will see in response to different actions.
192. So the different columns here show predicted future images in response to different motor commands.
193. This is quite a while back this is seven years ago.
194. So you can see that the predictions here are not of very high quality, but they capture the gist of what the robot is trying to do.
195. And they can be used to control objects.
196. So you can tell it, move this particular object marked in red to the green location.
197. It'll imagine the movement, and then it will actually actuate the arm to move the object in that way.
198. So predictive models can allow you to solve new tasks.
199. You can use this as a very powerful tool for emergent behavior.
200. You could, for example, command the robot to move some objects, and it might figure out that it needs to pick up a tool to move those two objects together.
201. Here's another tool use example here.
202. It figures out that that L-shaped tool can slide the blue object.
203. And here, there's an emergent tool use scenario where it figures out that to move these two pieces of trash, the water bottle makes for a nice improvised tool.
204. And predictive models have really come a long way.
205. So in recent years, we've been able to do a lot better with modern advances in general modeling.
206. This is a diffusion-based video prediction model that is being used to synthesize clips of driving videos.
207. The first three frames here are real.
208. The remaining frames are actually synthesized.
209. And you can see that the model will actually produce realistic camera movement.
210. It will introduce new objects as the car turns.
211. It will even predict the motion of the other cars with some reasonable fidelity.
212. In these examples, by the way, the left video in each pair is the real one.
213. The right one is the synthetic one.
214. And here, the same model is being run on robotic videos similar to the ones that I showed before, just so you can see the contrast from 2017 to 2022.
215. You can see that now the arm is clear and crisp.
216. The objects move in realistic ways and so forth.
217. There's also a lot of interesting progress, especially in the past year, on leveraging advances in pre-trained models.
218. So when we do reinforcement learning, we typically don't have to do it from scratch.
219. What we could do is we could use a model pre-trained on large amounts of Internet data and then use it for control.
220. This is actually an imitation learning example.
221. It doesn't do RL.
222. It actually does direct imitation, but it is doing a learning-based control.
223. This is the RT2 model, which uses a first a language model that is pre-trained on language, then a visual language model that uses that language model to process Internet images for things like question and answering, like what is happening in the image.
224. Let's say it's a great doggy walking down the street.
225. So now the model understands pictures.
226. It understands text.
227. And then that model is further fine-tuned to output robot actions so that when it's told what should the robot do to pick up the chips, it'll output the numerical values for the actions that will actually pick up the chips.
228. So now it can bring in knowledge that it learned from the Internet to perform this task more effectively.
229. Here are some examples of the kind of intelligence tests that this model can pass.
230. So it can be told to move the banana to the bottle.
231. The robot data has examples of moving bananas, but to understand what it means to move it to the bottle, it has to leverage Internet data.
232. Here it's asked to solve a math problem by putting the banana on the answer to the math problem.
233. Here it's told to put the strawberry into the correct bowl.
234. To figure out what correct bowl means, it needs to recognize the fruits in each of the bowls and figure out that the strawberry bowl is in fact the correct one.
235. And here are some more examples.
236. Pick up an object that is different from all the other objects.
237. Now it knows how to pick up objects in the robot data, but it doesn't know what different from all the others means from that.
238. But that has to leverage Internet data, and it figures out that the bar is the different object because all the other objects are bottles.
239. It can understand instruction in other languages, even though the robot data is only annotated in one language and so on.
240. Okay, so these are some examples of the kinds of problems that we might study in the context of learning-based decision making, besides the 4RL problems.
241. But to conclude this lecture, I want to end on maybe a somewhat more grandiose point.
242. I want to come back to this question, how do we build intelligent machines, and really argue that the basic building blocks of DeepRL might be very good building blocks for answering this question.
243. This is of course a controversial statement, I don't expect everybody to agree with me on this, but this is a big part of why I'm excited about this topic, and I hope to convey some of that excitement to you.
244. So, imagine that you have to build an intelligent machine, something as intelligent as a person.
245. Where would you start?
246. Well, in the olden days, the way we would think about this is that maybe we need to understand the brain, and the brain has a lot of parts, so let's understand what those parts are, figure out how each of them work, and then write computer programs to emulate the behavior of each of those parts.
247. Of course, our modern understanding of the brain is more advanced than what it was in the 19th century, but parts of the brain more closely reflect their actual function.
248. But this is still a very difficult problem, because each of the parts is very complex, and if we have to do a bunch of programming to code up the behavior of each of the parts, and do a bunch more coding to wire them together, we might be at this for a very long time.
249. That might just be a very, very difficult way to implement an intelligent machine.
250. It might actually take a lot more intelligence on our part than we actually have.
251. So, if we hypothesize that learning might be the basis of intelligence, that might actually offer us a much easier way to address this problem.
252. And here's an argument for why learning might be the basis of intelligence.
253. There are some things that we can all do, like walking.
254. So, it might be reasonably argued that maybe those things are sort of built into our brains somehow.
255. But there are also some things that we can only learn, like driving a car.
256. Clearly, driving a car is not built into our brains, because cars weren't around when our brains evolved.
257. And we can learn a huge variety of things, including very difficult things.
258. So, therefore, our learning mechanisms are likely powerful enough to do everything that we associate with intelligence.
259. It may be that in practice we don't actually use our learning mechanisms for some things, like walking, but we might hypothesize that maybe they're powerful enough that if we didn't have those things built in, we could figure it out anyway.
260. That may or may not be true, but I think there's a pretty good reason to believe this might be true.
261. It might still be very convenient to hard-code a few really important bits, but let's not get distracted by that part.
262. We can further hypothesize that not only is learning the basis of intelligence, but in fact maybe there's actually a single learning procedure that underlies all that we associate with intelligent behavior.
263. Now, that's a more radical statement.
264. It basically says that the way that we learn how to see and the way that we learn how to talk and the way that we learn how to hear is at some level the same.
265. Instead of having an algorithm for every module, maybe we have a single flexible algorithm that placed in the right context implements all of the modules, everything that we need in the brain.
266. And there's some circumstantial evidence to indicate that this might in fact be the case.
267. So, for example, these are some slides borrowed from Andrew Ng.
268. You can build an electrode array that you can put on your tongue, attach that array to a camera, and learn how to perceive visual percepts through your tongue.
269. You can take an animal, a ferret, you can disconnect the optic nerve from the visual cortex and plug it into the auditory cortex, and after a while the ferret will regain some degree of visual acuity, which means that its auditory cortex can essentially learn to process visual signals.
270. So these things kind of indicate that perhaps there's a degree of generality or homogeneity to the brain, for the neural cortex, such that it can adapt to whatever sensory input is provided, which might indicate that there's one algorithm.
271. And if there is one algorithm, what does this one algorithm need to be able to do?
272. Well, it needs to interpret rich sensory inputs, and it needs to choose complex actions.
273. And to do both of those things, we need large high-capacity models, because that's the only way we know how to deal with rich sensory inputs, and we need reinforcement learning, because that's the mathematical formulas we use to take actions.
274. So, why deep reinforcement learning?
275. Well, the deep part provides us with scalable learning from large complex data sets, and the reinforced learning gives us the optimization, the ability to take actions.
276. The combination of learning and search.
277. Deep is great for learning.
278. Reinforced learning is the way that we do the search.
279. And in fact, there is some evidence in neuroscience for both these things.
280. There's evidence that the kinds of representations acquired by deep neural networks have some statistical similarity to representations that are observed in the brain.
281. That doesn't mean that the brain works the same way that deep nets do, it just means that at some level, when you process lots of data and extract suitable representations, they end up looking similar.
282. Which could have more to do with the fact that a large enough learning machine just pulls out those patterns in the data, because that's what the data is made of.
283. Or it could say something about deep learning, that's, I think, a much harder question to answer, but the evidence suggests that some kind of representational similarity exists for visual percepts, for auditory features, and even for the sense of touch.
284. The experiments done to ascertain this are actually a little bit creative, where the brain signals indicating the kind of features that, in this case, monkeys use for touch, are obtained from recordings from monkey neurons.
285. The deep learning experiment is done by actually taking a glove dusted with white dust, getting a person to touch objects, and then using a deep neural network to discover patterns in the dust patterns on the glove.
286. So, interesting experiment, suggests that maybe the statistical properties of features extracted by sufficiently powerful learning machines resemble the features in the brain.
287. And there's plenty of evidence in favor of reinforcement learning as at least one of the mechanisms underlying decision making in humans and animals.
288. In fact, reinforcement learning actually emerged as a study of animal intelligence, but we know now from evidence that percepts that anticipate reward become associated with similar firing patterns as the reward itself, which is exactly what we would expect from a temporal difference learning process.
289. The basal ganglia appears to be a kind of reward system, and that model-free RL-like adaptation is often a good fit for experimental data of animal adaptation.
290. Although not always.
291. But the picture is not complete, right?
292. So all of these bits of circumstantial evidence might suggest that the tools of deep learning and reinforcement learning might be good tools for tackling the problem of intelligence.
293. But the problem is clearly not solved.
294. We have great methods that can learn from huge amounts of data by using deep learning, we have great optimization methods for RL, we don't yet have amazing methods that both use data in RL.
295. RL has been made much more scalable in recent years, it can tackle things like real-world robotics problems, but the kind of huge-scale language model and generative modeling applications still primarily use supervised learning, so there are still some algorithmic building blocks that are necessary.
296. And furthermore, humans learn incredibly quickly, whereas deep RL methods typically require large amounts of data.
297. And humans reuse past knowledge, whereas transfer learning in RL is still an open problem.
298. It's not always clear what the reward function should be, and it's not always clear what the role of prediction should be.
299. It seems like these methods can be very powerful, but how do they fit in with model-free methods?
300. Are they just different things, or can they be reconciled in some way?
301. So all of these question marks, I think, give us ample space for additional research that we can do in this area, and perhaps if the tools of deep learning and reinforcement learning are the right tools for building enormously powerful artificial intelligence systems, then maybe studying these questions can allow us to make some headway on that problem.
302. And ultimately, I think that we can get away from this picture of thinking of intelligence systems as a collection of modules to implement, and instead as a very elegant and simple framework where we have a general learning algorithm that can figure out whatever problems is posed to it.
303. In fact, this idea is not by any means new.
304. It's not something that was created in the 21st century.
305. It's not even something that was created for deep learning or even in the age of machine learning.
306. Here's a quote that I think very nicely exemplifies this perspective.
307. Instead of trying to produce a program to simulate the adult mind, why not rather try to produce one which simulates the child's?
308. If this were then subjected to an appropriate course of education, one would obtain the adult brain.
309. Who said this?
310. Alan Turing.