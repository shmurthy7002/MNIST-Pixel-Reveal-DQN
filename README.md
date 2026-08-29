So my goal was to create an agent that plays the following game:

agent is staring at black screen, gets to reveal 1 pixel, gets to see how bright that pixel was, agents gets to reveal 1 more pixel, gets to see how bright it was.
repeat the above many times (do 60 pixel reveals).

finally, the agent makes a guess at what the number is, hopefully it guesses right!

I initially decided to tackle this with a DQN to reveal the pixels, and then a pretrained classifier that looks at the pixels that have been revealed and make the final guess.
I quickly realized I needed tricks to make training stable and proceeded to add experience replay, target networks, Huber loss, parallel environments, gradient norm clipping, 
and this last part isn't related to making DQN more stable but an adam optimizer.

I am overall super duper proud of how this whole project has come along so far and I have honestly learned so much through working through all of the code for this myself, it forced
me to really ensure that I understood 100% of the ML theory inside out.

Right now it achieves pretty dang good accuracy after training (80%ish), but it learned a kind of goofy policy and I want to make it better (it learned to spread out its guesses, 
instead of learning to trace the numbers as I had hoped). If I have time I'll make it better, I think there are 2 major things that could be improved:
1. let the dqn itself also make the prediction for what the number is (no external pretrained oracle)
2. give the DQN some way of understanding spatial relationships between pixels (like with a CNN or an attention block)

btw: I did use ai for coding the visualization tools and diagnostics at the very bottom of dqn.ipynb, and I left a comment that says when that started. Everything else was done by yaboi
