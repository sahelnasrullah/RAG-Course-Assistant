# Project Summary (Part 1)

## Project Title

A Retrieval-Augmented Generation Assistant for Answering Questions about University Course Documents

## Group Members

- Ahmad Saloukha
- Sahel Nasrullah
- Mohamed Rami Karkour

## Problem Definition

Students need to find specific facts about their courses, for example how many
assignments a course has, how many credits each examination part is worth,
what the grading criteria are, and what the learning outcomes are. This
information is spread over course syllabi, course PDFs, study regulations and
learning-platform pages. Course documents are long, formatted differently and
often written in formal language. Finding one fact means opening several PDFs and
reading through them by hand.

This affects every student, and especially new students and international
students who are not yet familiar with the structure of Swedish course
documents. General-purpose LLMs are not a reliable substitute. They have not
seen the student's specific course document, so they tend to guess ("hallucinate")
numbers such as credits and number of assignments, and these errors matter for
planning one's studies. The main challenges are: (1) extracting information from
PDFs with tables and varying layouts, (2) retrieving the right passage from a
long document, and (3) making the model answer only from the document and say
so when the answer is not there.

## Motivation and Project Objective

We chose this problem because it is a real, daily need for students, including
ourselves. A Retrieval-Augmented Generation (RAG) approach fits well. The
answers must be grounded in a specific, changing set of documents, and
fine-tuning a model for every course document would be impractical. RAG lets the
system use new documents without retraining, and it can show which passage an
answer came from, so the student can verify it. This reduces hallucination
compared with asking an LLM alone.

The objective is to build and evaluate a simple RAG system. The user gives it a
course PDF and asks a question in natural language. For example, for the
question "How many assignments does this course have?" the system should answer
that there is one Assignment worth 3 credits (plus a separate Course Project
worth 3 credits), based on the examination section of the course syllabus.
We will:

1. Build a pipeline that loads PDFs, splits them into chunks, embeds them,
   stores them in a vector database and retrieves relevant chunks for a query.
2. Use an open-source LLM to generate answers that use only the retrieved
   context.
3. Evaluate the system on a set of questions with known answers, measuring
   answer correctness and faithfulness to the source. We will compare it with
   an LLM without retrieval to show the benefit of RAG.

## Dataset / Knowledge Source

The knowledge source is a collection of publicly available documents from
Blekinge Institute of Technology (BTH):

- **Course syllabi** (kursplaner) for BTH courses, which state learning
  outcomes, examination parts, credits and grading. These are published on
  bth.se. As our primary source we use the official syllabus of this course,
  *DV2658 Generative Artificial Intelligence* (6.0 credits, Dnr
  BTH-4.1.14-0213-2025, applies from 28 February 2025). Its "Assessment and
  grading" section lists the examination parts (Assignment, 3 credits, grade
  A-F; Course Project, 3 credits, grade G-U), which gives concrete,
  checkable answers for questions such as "How many assignments does this
  course have?". We will add syllabi from a few other BTH courses to test
  how well the system generalises.
- **BTH regulations and study rules**, for example the *Study administrative
  rules for preparatory education and first- and second-cycle education*
  (https://www.bth.se/download/18.7a47adda195b297cd34f318/1743070838602/Studieadministrativa-regler_per_220124_en.pdf)
  and *Modes of Assessment and Assessed Components at BTH*
  (https://www.bth.se/download/18.7a47adda195b297cd34f23c/1743070580999/Modes_of_Assessment_and_Assessed_Components_at_BTH.pdf).
- **Course documents** from our own courses (course PM, assignment descriptions),
  used only for local testing.

These sources are suitable because they come from the institution itself and
are therefore trustworthy. They contain exactly the kind of questions students
ask (credits, examination structure, rules). They are text-based PDFs, so they
are easy to process. They also contain tables and formal language, which makes
retrieval a realistic challenge. We will write a test set of about 30-50
questions with answers that we check manually against the documents.
