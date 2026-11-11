# A Old Habit

I have hobbies (not problems, I promise). And one of those hobbies, particularly when I was on exchange in the States, included the occasional (or not so occasional) visit to the casino to play my favourite card game: Blackjack.

It was my time on exchange at UC Davis, California where I really found my love for playing blackjack in real casinos, and I did some (definitely not obsessively) studying of the game and practicing at home before going out and trying my luck. It was an intimidating experience at first, but definitely one of my favourite thrilling past times after I got used to the casino environment. 

Anyway, I really love blackjack and although the statistics and probabilities are definitely already well documented, and programs similar to what I want to create exist, I figured it would be a fun project for me to practice some programming. Afterall, I think this can be a nice piece of work that'll not just be a piece of portfolio for job interviews, but a project that I can be quite proud of and give me another excuse to really do a deep dive into the game statistics and EV.

I figure I'll edit this README again, but for now, the plan is to explore some statistical methods and implement an Object Oriented system with python. Before I begin, I have some semblance of a plan which involves multiple iterations, each one building on the previous one, until I have something that's close to the final product I have in mind. How many iterations? I have no idea, but I'll definitely be starting from the ground up with the basics and fundamentals. It's been a while.

## How to play

Python 3.10 or newer. Nothing to install to play. Every iteration lives in its own
folder (`iteration_1`, `iteration_2`, ...) with the same layout, and the way to run
each one is identical.

From inside an iteration folder:

    python -m blackjack

In Google Colab, clone the repo, then run this in a cell (run below is for iteration 1 specifically):

    !git clone https://github.com/kualaS3/A-Old-Habit.git
    %cd A-Old-Habit/iteration_1

    from blackjack.cli import main
    main()

At each decision you type a command and press enter. Each iteration's own README
lists the commands that version supports.

Tests: `pip install pytest`, then run `python -m pytest` inside the iteration folder.
