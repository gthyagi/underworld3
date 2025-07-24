# JOSS Checklist

Following the [Submission requirements](https://joss.readthedocs.io/en/latest/submitting.html#submission-requirements)

## Repository format / content

    - [ ] README.md
      - [x] Links to uw web page
      - [x] Binder launch link for Quick-start
      - [x] Acknowledgements

    - [x] paper.md
    - [x] paper.bib
    - [x] Install GitHub workflow
    - [x] Validate workflow
    - [x] Open source licence file
    - [?] Publication branch (this is acceptable, must be kept up to date on submission).
    - [x] Check Insights / Community Standards for GitHub repo / put into line with UW2
    - [x] Check authors and .zenodo creators align

    - [x] Quickstart Guide for users (deployable on binder)
      - [x] Installation details
      - [x] Notebook examples (ipynb)
      - [x] Links to API docs
      - [x] Links to Github
      - [x] Links to Underworld Community

## Clean up checklist

    - [x] Audit GitHub issues / clear where possible
    - [x] Clear outdated PRs
    - [x] Merge dev into main
        - [x] create joss-submission branch from main at submission time
        - [x] joss-submission branch may need restructuring in order to build (e.g. paper in the root directory)

## Software checklist

Can I submit ?
    - [x] The software must have an obvious research application.
    - [x] You must be a major contributor to the software you are submitting, and have a GitHub account to participate in the review process.
    - [x] Your paper must not focus on new research results accomplished with the software.

The software associated with your submission must:
    - [x] Be stored in a repository that can be cloned without registration.
    - [x] Be stored in a repository that is browsable online without registration.
    - [x] Have an issue tracker that is readable without registration.
    - [x] Permit individuals to create issues/file tickets against your repository.

## paper.md validation

Preview the processed pdf:

**Note:** this is amazingly slow on macOS but don't give up, it will work eventually.

```bash
docker run --rm -it  -v $PWD:/data -u $(id -u):$(id -g) openjournals/inara -o pdf,crossref   joss-paper/paper.md
```
