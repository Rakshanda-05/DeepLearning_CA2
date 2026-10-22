"""
Dataset Generation and Preparation Module for Cyberbullying Detection.
Generates a realistic, well-balanced dataset of social media comments labelled as:
- 0: Non-Cyberbullying (Constructive, Neutral, Friendly, Supportive)
- 1: Cyberbullying (Insults, Threats, Harassment, Hate Speech, Shaming)
"""

import os
import random
import pandas as pd
import numpy as np

def generate_synthetic_cyberbullying_dataset(output_path: str, num_samples: int = 3000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    # Base templates for Non-Cyberbullying (Label 0)
    non_cyberbullying_samples = [
        "Great job on this presentation! Your insights were very helpful.",
        "I really appreciate you taking the time to explain this concept to everyone.",
        "Good game everyone, that was a tough match but we played well as a team.",
        "Happy birthday! Wishing you lots of joy and success this year.",
        "Can someone please explain how the loss function behaves in backpropagation?",
        "I disagree with your conclusion on this topic, but your methodology is sound.",
        "Thank you so much for the quick response and for helping me troubleshoot.",
        "You worked really hard on this project and it definitely shows in the quality.",
        "Let's meet up at the library around 4 PM to review the study guide.",
        "I hope you have a wonderful and restful weekend after this busy week.",
        "This tutorial helped me understand deep learning much better, thank you!",
        "Don't worry too much about the exam, you prepared thoroughly and will do great.",
        "Congratulations on your new job offer! You truly deserve this achievement.",
        "That was a really creative solution to a tricky programming bug.",
        "Does anyone have recommendations for good resources on computer vision?",
        "I love how supportive this online learning community has been for newcomers.",
        "Your artwork is stunning, the color palette and lighting are fantastic.",
        "Let's keep discussions respectful and focus on the data presented.",
        "The food at that restaurant was delicious, thanks for the great recommendation.",
        "I found a small typo in section 3, other than that the report is flawless.",
        "Listening to podcast episodes about artificial intelligence during my commute.",
        "Remember to stay hydrated and take regular breaks while studying.",
        "It's completely okay to make mistakes when you are just starting to learn.",
        "Appreciate the constructive feedback, I will incorporate these changes immediately.",
        "The conference keynote speaker gave some truly inspiring insights today.",
        "We should collaborate on the next research paper if you are available.",
        "Wishing everyone a productive and successful semester ahead!",
        "Learning Python and TensorFlow has been challenging but very rewarding.",
        "Thanks for sharing your code repository, it made replication straightforward.",
        "Peaceful morning walk in the park before heading to the campus laboratory."
    ]

    # Non-cyberbullying generative patterns
    non_templates = [
        "I really think {} is doing a fantastic job with {}.",
        "Have you considered looking into {}? It could improve {}.",
        "Thank you for sharing your thoughts on {}. It gave me a new perspective.",
        "Good luck with your upcoming {}! You have prepared well.",
        "Let's work together on {} to make sure we achieve high accuracy.",
        "I agree with the points raised about {}, very well articulated.",
        "Can anyone share insights on {}? Trying to understand the practical applications.",
        "Huge shoutout to the team for completing {} ahead of deadline.",
        "That is an interesting take on {}. I'd love to hear more about your findings.",
        "Always happy to help anyone who is struggling with {}."
    ]
    non_fillers_a = ["the project lead", "this developer", "our group", "the student committee", "the research team", "the professor", "the contributor", "the speaker"]
    non_fillers_b = ["the neural network architecture", "the evaluation report", "the final submission", "the coding assignment", "the study notes", "the open-source repo", "the data pipeline"]

    # Base templates for Cyberbullying (Label 1)
    cyberbullying_samples = [
        "You are completely useless, nobody wants you here in this group.",
        "Get off the internet, your existence is an embarrassment to everyone.",
        "You are so stupid and ugly, go cry somewhere else loser.",
        "Shut up, your opinion is trash just like you.",
        "Nobody likes you and you have zero talent, just quit already.",
        "You are disgusting and pathetic. Stop posting your garbage photos.",
        "Why are you even alive? Literally nobody would miss you.",
        "You are the dumbest person I have ever encountered on social media.",
        "Go jump off a cliff, you worthless piece of trash.",
        "Look at this clown trying to sound smart, you have no brain at all.",
        "Everyone is laughing at how pathetic and incompetent you are.",
        "I hope you fail completely and ruin your entire future.",
        "You are an absolute freak, nobody will ever love or respect you.",
        "You should delete your account and never show your ugly face again.",
        "Stop embarrassing yourself you utter failure, you deserve all the hate.",
        "Nobody cares about your stupid feelings, go suffer alone.",
        "You are so toxic and nasty, everyone in the server hates you.",
        "Keep crying little baby, your misery makes my day so much better.",
        "You look like a monster, honestly feel sick looking at your picture.",
        "Total waste of oxygen, do us all a favor and disappear forever."
    ]

    # Cyberbullying generative patterns
    toxic_templates = [
        "You are so {} and {}, literally everyone hates you.",
        "Nobody cares about your {} opinion, you absolute {}.",
        "Stop posting your {} garbage, you have no {} at all.",
        "You should just {} because you are an embarrassment.",
        "Everyone knows you are a {} who will always be a {}.",
        "Get lost you {} loser, nobody wants to see your {} face.",
        "You are completely {} at this game, uninstall and quit forever.",
        "Why do you even exist? You are pure {} and useless.",
        "Look at this {}, imagine being this {} and still talking.",
        "I swear you have negative IQ, you {} disgrace to your family."
    ]
    toxic_adj = ["stupid", "pathetic", "worthless", "disgusting", "horrible", "ugly", "idiotic", "clueless", "repulsive", "brainless"]
    toxic_nouns = ["loser", "clown", "freak", "failure", "joke", "trash", "waste of space", "parasite", "fool", "scumbag"]
    toxic_verbs = ["disappear", "shut up", "quit", "leave", "get lost", "hide", "stop breathing", "delete your account"]

    rows = []
    
    # Add base seeds
    for text in non_cyberbullying_samples:
        rows.append({"text": text, "label": 0})
    for text in cyberbullying_samples:
        rows.append({"text": text, "label": 1})

    # Generate additional samples to balance and reach num_samples
    target_each = num_samples // 2
    
    # Generate non-cyberbullying
    while sum(1 for r in rows if r["label"] == 0) < target_each:
        tmpl = random.choice(non_templates)
        fa = random.choice(non_fillers_a)
        fb = random.choice(non_fillers_b)
        text = tmpl.format(fa, fb)
        # Add natural variation
        if random.random() < 0.3:
            text += f" Looking forward to seeing more progress on {random.choice(non_fillers_b)}."
        elif random.random() < 0.2:
            text = "Honestly, " + text.lower()
        rows.append({"text": text, "label": 0})

    # Generate cyberbullying
    while sum(1 for r in rows if r["label"] == 1) < target_each:
        tmpl = random.choice(toxic_templates)
        a1 = random.choice(toxic_adj)
        a2 = random.choice(toxic_adj)
        n1 = random.choice(toxic_nouns)
        n2 = random.choice(toxic_nouns)
        v1 = random.choice(toxic_verbs)
        
        # Format based on placeholders
        try:
            if tmpl.count("{}") == 2:
                text = tmpl.format(a1, n1)
            elif tmpl.count("{}") == 3:
                text = tmpl.format(a1, n1, a2)
            else:
                text = tmpl.format(v1)
        except Exception:
            text = f"You are a {a1} {n1}, go {v1} forever."
            
        if random.random() < 0.25:
            text += f" Nobody will ever care about a {n2} like you."
        rows.append({"text": text, "label": 1})

    # Shuffle dataset
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Dataset successfully generated and saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "cyberbullying_dataset.csv")
    generate_synthetic_cyberbullying_dataset(target_csv, num_samples=3200)
