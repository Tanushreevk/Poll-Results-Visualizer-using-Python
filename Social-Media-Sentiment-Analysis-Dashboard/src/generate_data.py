import pandas as pd
import random

# Basic components
subjects = ["app", "service", "product", "experience", "platform"]

positive_words = ["good", "great", "amazing", "excellent", "nice"]
negative_words = ["bad", "terrible", "worst", "awful", "poor"]
neutral_words = ["okay", "fine", "average", "normal"]

# Sentence templates
positive_templates = [
    "This {} is {}",
    "I really like this {}",
    "Very happy with this {}",
    "This {} works {}",
    "Absolutely love this {}"
]

negative_templates = [
    "This {} is {}",
    "I really hate this {}",
    "Very disappointed with this {}",
    "This {} does not work well",
    "Worst {} I have used"
]

neutral_templates = [
    "This {} is {}",
    "This {} is just okay",
    "Average experience with this {}",
    "This {} works fine",
    "Nothing special about this {}"
]

# Real-world mixed sentences (IMPORTANT 🔥)
mixed_sentences = [
    ("This app is good but slow", "neutral"),
    ("This service is not bad", "positive"),
    ("This product is not good", "negative"),
    ("Works fine but crashes sometimes", "neutral"),
    ("Great experience but support is bad", "neutral"),
    ("I like this app but it has bugs", "neutral"),
    ("Not the best but okay", "neutral"),
    ("Amazing features but very slow", "neutral"),
    ("Bad at first but now good", "positive"),
    ("Good but needs improvement", "neutral"),
]

data = []

# Generate positive data
for _ in range(250):
    template = random.choice(positive_templates)
    sentence = template.format(
        random.choice(subjects),
        random.choice(positive_words)
    )
    data.append([sentence, "positive"])

# Generate negative data
for _ in range(250):
    template = random.choice(negative_templates)
    sentence = template.format(
        random.choice(subjects),
        random.choice(negative_words)
    )
    data.append([sentence, "negative"])

# Generate neutral data
for _ in range(250):
    template = random.choice(neutral_templates)
    sentence = template.format(
        random.choice(subjects),
        random.choice(neutral_words)
    )
    data.append([sentence, "neutral"])

# Add mixed realistic data (KEY PART 🔥)
for _ in range(400):
    sentence, label = random.choice(mixed_sentences)
    data.append([sentence, label])

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "sentiment"])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove duplicates (important)
df = df.drop_duplicates()

# Save to CSV
df.to_csv("data/social_media_data.csv", index=False)

print("✅ Final realistic dataset created!")
print("Total rows:", len(df))