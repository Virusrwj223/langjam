import random
import sys

# Game constants derived from the AST
TITLE = "Brainrot Mental Sums: 67 Mode"
SETTING = "A neon study room where every correct sum gives you +vibes"
TONE = "absurd"
SEED = 67
PLAYER_HP_START = 5
MAX_TURNS = 25
SCORE_TARGET = 10  # threshold for win_condition score_threshold
ACTIONS = {"interact", "wait"}

random.seed(SEED)

def print_status(turn, hp, score):
    """ASCII render of current game state."""
    print("\n" + "=" * 40)
    print(f"{TITLE}")
    print(f"Turn: {turn}/{MAX_TURNS} | HP: {hp} | Score: {score}")
    print("=" * 40)

def main():
    hp = PLAYER_HP_START
    score = 0
    turn = 1

    while turn <= MAX_TURNS and hp > 0 and score < SCORE_TARGET:
        print_status(turn, hp, score)
        action = input("Choose action (interact/wait): ").strip().lower()
        if action not in ACTIONS:
            print("Invalid action. Try again.")
            continue

        if action == "interact":
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            correct = a + b
            try:
                answer = int(input(f"What is {a} + {b}? "))
            except ValueError:
                answer = None
            if answer == correct:
                score += 1
                print("Correct! +1 score.")
            else:
                hp -= 1
                print(f"Wrong! The correct answer was {correct}. HP -1.")
        elif action == "wait":
            print("You wait patiently... nothing happens.")

        turn += 1

    # Determine outcome
    print("\n" + "=" * 40)
    if score >= SCORE_TARGET:
        print("Congratulations! You reached the score threshold and won!")
    elif hp <= 0:
        print("Game Over! Your HP dropped to zero.")
    else:
        print("Game Over! You ran out of turns.")
    print(f"Final Score: {score} | Final HP: {hp}")
    print("=" * 40)

if __name__ == "__main__":
    main()
