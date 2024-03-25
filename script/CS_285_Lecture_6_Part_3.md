1. All right, next we're going to talk about some design decisions for actually implementing Actor-Critic algorithms.
2. So we'll start with the discussion of neural network architectures.
3. In order to actually instantiate these algorithms as deep RL algorithms, we have to pick how we're going to represent the value function and the policy.
4. So before, in the last lecture, we just had the policy to deal with.
5. Now we have to represent both of these objects.
6. And there are a couple of choices we could make.
7. So one very reasonable starting choice, and this is the one that I would recommend if you're just getting started, is to have two completely separate networks.
8. So you have one network that maps a state to the value, and then you have another completely separate network that maps that same state to the distribution over actions.
9. And these networks have nothing in common.
10. This is a convenient choice because it's relatively simple to implement, and it tends to be fairly stable to train.
11. The downside is it may be regarded as somewhat inefficient because there's no sharing of features between the actor and critic.
12. This could be a more important issue if, for example, you are learning directly from images, and both these networks are convolutional neural nets.
13. Maybe you would really want them to share their internal representations so that, for example, if the value function figures out good representations first, the policy could benefit from them.
14. In that case, you might opt for a shared network design where you have one trunk.
15. Maybe this represents the convolutional layers.
16. And then you have separate heads, one for the value and one for the policy action distribution.
17. This shared network design is a little bit more complicated.
18. It's a little bit more complicated.
19. But it's a little bit more complicated.
20. So you have two different layers.
21. The shared network design is a little bit harder to train.
22. It can be a little bit more unstable because those shared layers are getting hit with very different gradients.
23. The gradients from the value regression and the gradients from the policy gradient, they'll be on different scales.
24. They'll have different statistics.
25. And therefore, it might require more hyperparameter tuning in order to stabilize this approach.
26. But it can, in principle, be more efficient because you have these shared representations.
27. Now there is another important point that we have to discuss before we get an actual practical deep review.
28. And that is the question of batch sizes.
29. So as described here, this algorithm is fully online, meaning that it learns one sample at a time.
30. So it takes an action, gets a transition, updates the value function on that transition, and then updates the policy on that transition.
31. And both updates use just one sample.
32. Now, we know from the basics of deep learning that updating deep neural nets with stochastic gradient descent using just one sample is not going to be a good idea.
33. It's going to be a little bit more complicated.
34. But it's going to be a little bit more complicated.
35. So we're going to be looking at some of these steps.
36. And I'm going to start with the first step, which is the deep learning.
37. And then we're going to look at the second step, which is the learning of the deep neural nets.
38. We're going to increased it.
39. Now, here.
40. What we've got here is the students here.
41. And these students are going to be learning mobility systems that we've been taught how to use in the in the intraspect arena reviews classroom and encima.
42. And what we saw here is that companies have gone out of the verdure system.
43. What you're seeing in the video is the way you Давайте Okay.
44. am I going to explain the families here.
45. Why is that it happens because you can do it.
46. What is it here for?
47. What images is this designed for for people who have the device running This is the most basic kind of parallelized actor critic.
48. It's a synchronized parallel actor critic.
49. Instead of having just one data collection thread, instead of just running one simulator, you might run multiple simulators, and each of them will choose an action in step one and generate a transition.
50. But they're going to use different random seeds, so they'll do things that are a little bit different.
51. And then you will update in step two and step four using data from all of the threads together.
52. So the update is synchronous, meaning that you take one step in step one for each of the threads, then collect all the data into your batch and use it to update the value function, and then use it to update the policy synchronously.
53. And then you repeat this process.
54. So this will give you a batch size equal to the number of worker threads.
55. It can be a little bit expensive, right, because if you want a batch size of like 32, then you need 32 worker threads, but it does work decently well.
56. Now it can be made even faster if we make it into asynchronous parallel actor critic.
57. Meaning that we basically drop the synchronization point.
58. So now we have these different threads that are all running at their own speed.
59. And when it comes time to update, what we're going to do is we're going to pull in the latest parameters and we're going to make an update for that thread, but we will not actually synchronize all the threads together.
60. So just as soon as we accumulate some number of transitions, let's say we got 32 transitions from all the workers, we'll make an update.
61. Now the problem with this approach, of course, is that the actual transitions might not have been collected by exactly the same parameters.
62. So if one of the threads is lagging behind, maybe its transition was generated by an older actor, and then you will basically not actually update until you get transitions from faster threads, and those will be using a newer actor.
63. So in general, all of the transitions that you're pulling together into your batch, which is the first one here, may have been generated with slightly different actors.
64. Now they're not going to be too different because these threads aren't going to be running at such egregiously different rates, but there will be a little bit lagging behind.
65. So an obvious question to ask here is, well, is this kind of update, the asynchronous update, mathematically equivalent to the standard synchronous update?
66. And the answer is that it isn't, that you have this small amount of lag, which is similar to what you would get with asynchronous SGD.
67. But in practice, it usually turns out that making the method asynchronous, leads to gains in performance that outweigh the bias incurred from using slightly older actors.
68. The crucial thing here is slightly older, right?
69. Because the actors are not going to be too old.
70. If they're too old, then of course this won't work.
71. But as long as none of the threads hang up, then you'll be okay.
72. But this might get us thinking about another question.
73. Well, in the asynchronous actor critic algorithm, the whole point was that we could use transitions that were generated by slightly older actors.
74. If we can somehow get the transition that we're going to use from the same actor, and somehow get away with using transitions that were generated by much older actors, then maybe we don't even need multiple threads.
75. Maybe we could use older transitions from the same actor.
76. Basically, maybe we could use a history and load in transitions from that history and not even bother with multiple threads.
77. And that's the principle behind off-policy actor critic.
78. So the design of off-policy actor critic is that now you're going to have one thread, and you'll update with that one thread.
79. But when you update, you're going to use a replay buffer of old transitions that you've seen, and you will actually load your batch from that replay buffer.
80. So you're actually not going to necessarily use the latest transition.
81. You'll collect a transition, store it in the replay buffer, and then sample an entire batch from that replay buffer, maybe 32 transitions rather than just one, and update on that batch.
82. Now at this point, we have to modify the algorithm, because doing this naively won't work.
83. This batch that we loaded in from the replay buffer definitely came from much older policies.
84. So it's not like the asynchronous actor critic buffer that we saw before, where the transitions came from just slightly older actors and we could just ignore that.
85. Now it's coming from much older actors, and we can't ignore that.
86. We have to actually change our algorithm.
87. So when I say replay buffer, basically I just mean a buffer that contains transitions that we saw in prior timestamps.
88. The most straightforward way to implement a replay buffer is to implement it as a ring buffer, a first-in, first-out buffer, where you batch up, let's say, one million transitions.
89. I will say here that we will discuss replay buffer and replay buffer much, much more in a subsequent lecture.
90. So don't get too caught up on this for now.
91. It's just a buffer that stores all the data, all the experience you've seen so far.
92. And then, of course, we're going to form a batch for each of these updates by using previously seen transitions.
93. Okay, so let's see what this might look like in an off-policy actor critic algorithm.
94. We're going to take an action, as usual, from our latest policy.
95. Get the corresponding transition.
96. But then instead of using that transition for learning, we'll actually store it in our replay buffer.
97. Then we will sample a batch from that replay buffer.
98. So this notation denotes a set of n transitions, each of them indexed with i.
99. It might not even contain our latest transition.
100. So when we load this batch from the buffer, it might not contain that latest transition that we sampled, and that's okay.
101. And then we're going to update our value function using targets for each of these transitions in our batch.
102. So we have capital N transitions, which means we have capital N targets.
103. So we're going to compute the gradient of our loss averaged over the batch.
104. So n here is the batch size.
105. It's not the total buffer size, it's just the size of the batch.
106. So it might be 32 or 64.
107. Then we'll evaluate our advantage, again, for each of the samples in our batch.
108. And then we'll update our gradient, our policy gradient, by using that batch.
109. So now the policy gradient is also averaged over n samples.
110. And then we'll update our gradient, and then we'll apply the policy gradient like before.
111. So this algorithm is not going to work the way I described.
112. It's actually quite broken, and we have to do a bunch of things to fix it.
113. One thing that I would recommend as an exercise here is to pause the video, look at this algorithm, and try to guess where it's broken.
114. I'll tell you right now it's broken in at least two places.
115. Meaning that in at least two places in the pseudocode there's something that doesn't make sense.
116. Try to pause the video and find it, and then you can resume and I'll tell you what it is.
117. Okay.
118. So the first problem is that when you load these transitions from the replay buffer, is that when you load these transitions from the replay buffer, remember that the actions in those transitions were taken by older actors.
119. So when you use those older actors to get the action and compute the target values, that's not going to give you the right target value.
120. It'll give you the value of some other actor, not your latest actor.
121. And that is not what you want.
122. So formally the answer is that the target value is not the same as the target value.
123. The issue is that AI did not come from the latest π theta.
124. It came from some older π theta.
125. And therefore si prime also was not the result of taking an action with the latest actor.
126. And that's a problem.
127. The second issue is that for that same reason, because AI didn't come from the latest policy π theta, you can't compute the policy gradient this way.
128. Remember from the previous lecture it is very very important when computing the policy gradient that we actually get actions that were sampled from our policy, because this needs to be an expected value under π theta.
129. If that is not the case, we need to employ some kind of correction, such as importance sampling.
130. And we could actually do this with importance sampling, but it turns out there's actually a better way to do it for off-policy actor critic, which I will tell you about next.
131. But first, let's talk about fixing the value function.
132. So I'll first fix the problem in step 3, and then I'll fix the problem in step 5.
133. So to fix the problem in step 3, instead of working with value functions, let's instead think back to lecture 4, where we also introduced this notion of a Q function.
134. If the value function tells you the expected reward you will get if you start in state s and then follow the policy π, the Q function tells you the reward you'll get if you start in state s , then take action a and then follow the policy π.
135. Now notice here that there's no assumption that the action a actually came from your policy.
136. So the Q function is a valid function for any action.
137. It's just in all subsequent steps you follow π.
138. So what we're going to do to accommodate the fact that our transition s , a , s did not come from our latest policy π is that we will actually not learn v, but we will instead learn q.
139. So we will not keep track of ^{V}^π_ϕ, we will keep track of ^{Q} π ϕ.
140. It's going to be a different neural network.
141. We'll take in a state and an action and output a q value.
142. But otherwise the principle behind the update is the same.
143. So we're going to compute target values and then we will regress onto those target values.
144. It's just that now we'll give the action as an input to the Q function.
145. Another way to think about it is we can no longer assume that our action came from our latest policy π theta, so we'll instead learn a state action value function that is valid for any action so that we can train it even using actions that didn't come from π theta, but then query it using actions from π theta.
146. Now those of you that are paying attention might notice that there's a little bit of an issue here.
147. Because before I was learning v hat and I was using v hat in my targets.
148. And that's okay because I'm learning v hat so I have it available to me to use in my targets.
149. But now I'm learning q hat, but I still need v hat for my target values.
150. So where do I get that?
151. Well, remember that the value function can also be expressed as the expected value of the Q function where the expectation is taken under your policy.
152. So what we can do is we can replace the v in our target value with q, evaluate it at the action ai prime, except that ai prime now is not the action from our replay buffer.
153. ai prime is actually the action that your current policy π_θ would have taken if it had found itself in si prime.
154. So you'll actually sample si ai si prime from your replay buffer, but then you will sample ai prime by actually running your latest policy.
155. And you can do that because your policy is just a neural network.
156. You don't have to actually interact with a simulator to ask the policy what action it would have taken.
157. So it's a little trick that we're pulling here.
158. We're actually exploiting the fact that we have functional access to our policy so we can ask the policy what action it would have taken.
159. We can ask our policy what it would have done if it had found itself in this old state si prime even though that had never actually happened.
160. So then we get this action ai prime and we plug it into the q value.
161. And that gets us a target value that actually represents the value of the latest policy at this old state si prime.
162. That's really cool.
163. Okay, so we've resolved our issue with the value function.
164. Instead of learning v, we're going to learn q and we're going to exploit the fact that we can evaluate the value function with the expected value of the Q function under the policy.
165. Now, how are we going to deal with step 5?
166. How are we going to deal with a policy gradient?
167. Well, all we're going to do is we're going to use the same trick but this time we're going to use it for ai instead of ai prime.
168. So in order to evaluate the policy gradient we need to figure out an action sampled from the latest policy π_θ at the state si.
169. But of course we can do that.
170. We can just ask our policy what it would have done at the state si if it had the option to act there.
171. And we'll call this action ai π to differentiate it from ai.
172. So ai was actually from the buffer.
173. ai prime is what the policy would have done if it had been in the buffer state si.
174. And now we'll just plug in this ai π into our policy gradient equation and that's now correct because ai prime did in fact come from π_θ so this is in fact an unbiased estimator of expectations under π theta.
175. So remember, ai π here is not the action that we're going to be using.
176. So we're going to use this as an example of how we can evaluate the policy gradient at the state from the replay buffer.
177. It's the action sampled from your policy at the state from the replay buffer.
178. Now in practice when we do this kind of off-policy actor critic we don't actually use the advantage values.
179. We just plug in our ^{Q} directly into this equation.
180. We don't have to do it.
181. We could actually calculate advantages.
182. There's nobody stopping us from doing that.
183. But it turns out that it's very convenient to just plug in q values.
184. They have higher variance because they're not being baseline.
185. But higher variance is actually okay here.
186. Why is that?
187. Well it's because we don't need to interact with a simulator to sample these actions ai prime.
188. So it's actually very easy to lower our variance just by generating more samples of the actions without actually generating more sampled states.
189. So it doesn't require any simulation it just requires running the network a few more times.
190. So in practice we're actually okay with a higher variance here because in exchange we get a larger batch size and it's all good.
191. And it spares us the complexity of computing the advantage of step four.
192. So we're actually going to completely drop step four for off-policy actor critic algorithms and we'll use ^{Q} instead of a hat.
193. Which is still unbiased it just doesn't have the baseline.
194. So that gives us the more or less complete algorithm for off-policy actor critic.
195. What else is left?
196. Well there is still a little bit of an issue.
197. Because SI the state that we're actually using itself it didn't come from the state margin of the latest policy.
198. It came from the state margin of an old policy.
199. Unfortunately there's basically nothing we can do here.
200. So this is going to be a source of bias in this procedure and we'll just have to accept it.
201. The intuition for why it's not so bad is because ultimately we want the optimal policy on p θ of s but we get the optimal policy on a broader distribution.
202. So our replay buffer will contain samples from the latest policy as well as many samples from other older policies.
203. So the distribution is sort of broader than the one we want.
204. So we don't want to be on the states from our latest policy we just also have to be good on a bunch of other states which we might never visit.
205. So we're doing kind of extra work but we're not missing out on important stuff.
206. And that's the intuition for why this basically tends to work.
207. Okay so a few details here.
208. If you actually read some papers and I'll reference a paper here shortly that implement this procedure one of the things you'll notice is that often times there's much fancier things we can do for step four.
209. There's something called the reparameterization trick which I'll discuss in the second half of the course much later so don't worry about it for now but that can be a better way to estimate this integral.
210. There are also many fancier ways to fit the Q function and we'll discuss this in the next two lectures when we talk about Q learning.
211. So I described a very naive way to fit the Q function but there are actually better ways to do it.
212. If you want an example of a practical algorithm that builds on this idea check out the algorithm called soft actor critic.
213. This is actually one of the most widely used actor critic methods today.
214. Although the online value based actor critic methods are more classical the off policy Q value based actor critic methods are more commonly used.
215. And we'll also learn about algorithms that do this kind of thing with deterministic policies later.
216. So this is for a stochastic actor later on when we talk about Q learning we'll actually revisit off policy actor critic methods also with deterministic actors.