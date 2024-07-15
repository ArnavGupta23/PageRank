import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create a dictionary object to hold the probability distribution
    probability_distribution = {}
    # Get  list of linked pages by the current page
    linked_pages = corpus[page]

    # If the page has no outgoing links, act as it has links to all pages including itself
    if len(linked_pages) == 0:
        for p in corpus:
            probability_distribution[p] = 1 / len(corpus)
    
    # Calculate the probability distribution
    else:
        for p in corpus:
            probability_distribution[p] = (1 - damping_factor) / len(corpus)
        for p in linked_pages:
            probability_distribution[p] += damping_factor / len(linked_pages)

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # Initialize the PageRank dictionary object
    pagerank = {page: 0 for page in corpus}
    # Select a ramdom page at first
    sample = random.choice(list(corpus.keys()))
    # Estimate PageRank by incrementing counts based on transition probabilities.
    for i in range(n):
        pagerank[sample] += 1
        distribution = transition_model(corpus, sample, damping_factor)
        sample = random.choices(list(distribution.keys()), weights=distribution.values(), k=1)[0]

    # Normalize the results so that the PageRank values sum = 1
    total_samples = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total_samples

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Number of pages in the corpus
    corp_len = len(corpus)
    # Initialize the PageRank values to 1/N for each page
    pagerank = {page: 1 / corp_len for page in corpus}
    # Copy of the PageRank values for the update process
    new_pagerank = pagerank.copy()

    change = True
    while change:
        change = False
        for page in pagerank:
            total = 0
            for p in corpus:
                # Sum the PageRank contributions from pages that link to the current page
                if page in corpus[p]:
                    total += pagerank[p] / len(corpus[p])
                # If the page has no outgoing links, treat it as linking to all pages
                if len(corpus[p]) == 0:
                    total += pagerank[p] / corp_len
            # Update the PageRank value for the current page
            new_pagerank[page] = (1 - damping_factor) / corp_len + damping_factor * total

        # Check convergence
        for page in pagerank:
            if abs(new_pagerank[page] - pagerank[page]) > 0.001:
                change = True
        pagerank = new_pagerank.copy()

    # Normalize the results so that the PageRank values sum to 1
    total_pagerank = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total_pagerank

    return pagerank


if __name__ == "__main__":
    main()
