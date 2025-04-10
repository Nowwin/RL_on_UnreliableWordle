import random

def select_random_words(input_filename, output_filename, num_words=7000):
    # Read the words from the input file, assuming whitespace (or newlines) separation.
    with open(input_filename, 'r', encoding='utf-8') as file:
        content = file.read()
    words = content.split()

    # Verify there are enough words in the file.
    if len(words) < num_words:
        raise ValueError(f"The file contains only {len(words)} words, which is less than {num_words}.")

    # Randomly select the required number of words without replacement.
    selected_words = random.sample(words, num_words)

    # Write each selected word on a new line to the output file.
    with open(output_filename, 'w', encoding='utf-8') as file:
        for word in selected_words:
            file.write(f"{word}\n")

if __name__ == '__main__':
    input_file = 'wordList.txt'         # The source file.
    output_file = 'wordList2.txt'   # The output file that will be created.
    select_random_words(input_file, output_file, 7000)
