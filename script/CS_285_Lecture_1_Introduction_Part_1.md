1. Let's say that you'd like to build a system to enable a robot like this to pick up objects.
2. So this is what the robot sees.
3. It sees images from its camera.
4. And the goal is to output coordinates in space using some kind of machine that you're going to build that will allow it to pick up objects successfully.
5. Now this is actually a pretty tricky problem to solve because while you might think that all you have to do is localize where the objects are in the picture and just output their position, in reality the right way to pick up an object actually has a lot of special cases and exceptions that you need to take into account.
6. So if you want to really understand the problem and design a solution manually, maybe rigid objects are fairly straightforward to pick up.
7. You just put the fingers on either side.
8. But if the object is awkwardly shaped and has a complex mass distribution, then you need to really make sure you pick it up closer to the center of mass so that it doesn't fall out of the gripper.
9. And if the object is soft and deformable, then an entirely different set of strategies might be more important.
10. Like pinching it.
11. So anytime we have a situation that has so many special cases, exceptions, and little details, it makes it very appealing to use machine learning.
12. So it would be really nice to try to set this up as a machine learning problem where instead of having to manually engineer all these little exceptions, you could just run some kind of general purpose machine learning procedure, maybe with convolutional neural networks, to extract suitable grasp locations from an image automatically.
13. The trouble is that the standard tool we have in supervised learning don't make this very easy, because they require us to somehow obtain a data set consisting of pairs of images and suitable grasp locations.
14. But the problem is that even people can't necessarily determine grasp locations very well, because they're really a property of the physical interaction between the robot and its environment, not necessarily something that is very well informed by human intuition.
15. To put it simply, we don't have a lot of experience picking things up with robot fingers.
16. So can we somehow use machine learning but avoid the need to manually supervise this process?
17. Well, what if we actually get the robots themselves to collect a lot of trials to attempt different grasps and see what works and what doesn't work?
18. That in essence is the main idea behind reinforcement learning, and the methods that we'll discuss in this course will in some ways address different methods for tackling this type of problem.
19. So in a reinforcement learning setting, we wouldn't try to manually specify, in this case, where the robots should grasp objects.
20. Instead, the machines themselves will collect the data set that doesn't necessarily consist of good examples, but examples that are labeled with our outcome.
21. So it'll be images, what the robot did, and whether that led to a failure or success.
22. More generally, we would refer to this as a reward function.
23. The robot would be rewarded for success and not for failure.
24. And then this would be used in combination with a reinforcement learning algorithm.
25. A reinforcement learning algorithm is doing something very different from a supervised learning algorithm.
26. It's not just trying to copy everything that's in the data.
27. It's trying to use these success-failure labels, these reward labels, to figure out what it should do in order to maximize the number of successes or to maximize the reward.
28. And then perhaps we could get a policy that's actually better than the average behavior that the robot carried out while it was collecting data, that it actually uses that experience to improve upon what it would typically do.
29. Okay, so that's kind of the bigger picture.
30. But now let's put this in the context of what has been happening lately in artificial learning.
31. What are some recent advances we've seen in AI?
32. Well, the last few years have been very active in artificial intelligence.
33. We've seen pretty impressive advances, for example, in the ability of AI systems to generate pictures in response to a textual prompt.
34. You could, for example, get a diffusion model where you can tell it, please provide a vibrant portrait painting of Salvador Dalí with a half-robot face, and it will actually generate a plausible-looking picture showing that.
35. We could get language models that can carry out conversations that can tell you jokes about cows going to study bovine sciences at Harvard.
36. You can get large language models that act as assistants that can explain jokes that can even answer complex coding prompts.
37. And even outside of the kind of standard generative modeling applications, we've seen a lot of interesting results, for example, in biological sciences, where you can get generative models that will produce proteins that bind to certain kinds of viruses.
38. So data-driven AI has really advanced tremendously, and we've seen a lot of advances from image generation to text to all sorts of other areas.
39. A lot of these advances that have been very much in the news in the last few years are based on, in some sense, a very similar idea to the supervised learning approach that I presented as kind of a straw man in my discussion of the robotic example from before.
40. The principle behind the image generation model, the language models and many of these other settings, is based on essentially a kind of density estimation, estimating p(x) or conditional density estimation P of Y given X.
41. So for language models, typically estimate the distribution of natural language sentences.
42. The image generation models might be conditional distributions over images conditioned on that prompt.
43. But it's a very similar kind of idea.
44. And in both cases, these are really just massively scaled up versions of the kind of density estimation that we learn about in statistics class.
45. And of course, a very important thing to remember when you're doing density estimation, when you're doing essentially supervised learning, is that what you're learning about is the distribution in the data.
46. And that makes it very important to think about where the data actually comes from.
47. So if the data consists of large amounts of images mined from the web, for example, and those images are labeled with textual prompts, then what you're really learning about is the kind of images that people put on the web, the kind of pictures, for example, they might photograph.
48. In the case of text, you're learning about what people tend to type on keyboards.
49. Now, these are very good things to learn from if your goal is to generate content that is similar to what humans would have generated.
50. If your goal is to generate the kinds of paintings that humans would have drawn or the kind of text that humans would have written, and that can give you a very powerful capability.
51. But of course, it's not the only thing that we want from our autonomous systems.
52. So what does reinforced learning do differently?
53. Well, before we talk about that, we need a little bit of kind of historical background on what modern reinforced learning is and where it came from.
54. And really, modern reinforcement learning traces its lineage to two previous disciplines.
55. The first one, which is the one that's actually called reinforcement learning, actually has its roots in psychology, and particularly in the study of animal behavior.
56. So this is a photograph of Skinner, who was a very well-known researcher who studied the behavior of animals in response to various kinds of reinforcement.
57. And much of the work stemming from that line of research forms the bedrock of the kind of reinforcement learning that we do today in computer science, which models an agent as interacting with its environment and adapting to its environment in response to rewards.
58. But there's a different kind of pedigree that also heavily influences modern reinforcement learning, which has to do with controls, optimization, and also has its roots in things like evolutionary algorithms.
59. This is a video from 1994 produced by Carl Sims that shows an optimization procedure, which Sims did not call reinforcement learning he referred to it as evolution, but had somewhat similar principles, that was used to optimize both the form and the behavior of these virtual creatures.
60. So these virtual creatures would do things like locomote, swim, run around, they would even fight each other, and their behaviors would be optimized, they would be emergent.
61. So this is very different from the kind of machine learning that we think about today, where the goal is to reproduce the behavior of humans.
62. Here the behavior, the goal was to actually produce behaviors that did not need to be designed by humans.
63. And if we fast forward a couple of decades, we can see with more sophisticated algorithms for, in this case, automated optimization and control, this is a result by Yuval Tassa that shows a humanoid kind of simulated robot automatically figure out how to do things like walk and run and so on.
64. So these two disciplines together actually influence the study of modern deep reinforcement learning, which could be thought of as the combination of large-scale optimization, with the kinds of algorithmic ideas and foundations derived from classical reinforcement learning.
65. And that's actually very powerful, because once we take those classical reinforcement learning ideas and we scale them up with the tools of modern computational optimization, then we can get very powerful emergent behaviors.
66. So many of you probably know about AlphaGo.
67. There was a very dramatic moment in the AlphaGo championship match, that was sometimes referred to as Move 37, where the AlphaGo system performed a move that experts watching the game were very surprised by.
68. And it was surprising because this was not the kind of move that human players would have likely made in these kind of situations.
69. It was an emergent behavior.
70. Now, the generative AI results that we've seen in recent years are very impressive precisely because they look like something that a person might produce.
71. The pictures look like pictures that a person might draw.
72. The most impressive results of reinforcement learning are actually impressive precisely because no person had thought of it.
73. What makes the results in AlphaGo so interesting to us is the emergence, the fact that an automated algorithm could discover a solution that goes beyond what people would do.
74. And this is really, really important if we're going to take the study of AI seriously because we probably won't get the kinds of flexible intelligence that we associate with humans if we merely copy human behavior.
75. We really have to figure out how to get algorithms that discover solutions that are the best solution for the task, rather than merely the solution that a person would have taken because then, when placed in novel situations, they'll actually respond intelligently.
76. So, this is the motivational program I'm going to talk about.
77. And in the remainder of this lecture, I'll take you through the structure of the course and then describe a little bit more some of the motivations for why we should study deep reinforcement learning today.