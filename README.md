# Orthogonal Gradient Descent (OGD) for Continual Learning

This repository explores the concept of Orthogonal Gradient Descent (OGD) as a method to mitigate catastrophic forgetting in deep neural networks during continual learning scenarios. Catastrophic forgetting is a significant challenge in artificial intelligence systems, where models tend to forget previously acquired knowledge when subsequently trained on new tasks. OGD aims to address this by carefully controlling the direction of gradient updates in the parameter space, ensuring that the network's performance on prior tasks remains largely unaffected while still allowing for effective learning of new tasks.

## Project Structure

The repository is organized into two main directories:

  * **`pdfs/`**: Contains academic reports and a lecture document detailing the theoretical background and experimental findings.

      * [`mini-report.pdf`](https://www.google.com/search?q=uploaded:mini-report.pdf): A concise report focusing on catastrophic forgetting in a simple sequential learning setup using the MNIST dataset.
      * [`report.pdf`](https://www.google.com/search?q=uploaded:report.pdf): A comprehensive report on Orthogonal Gradient Descent for Continual Learning, discussing its methodology and experimental results.
      * [`lecture.pdf`](https://www.google.com/search?q=uploaded:lecture.pdf): "Orthogonal Gradient Descent for Continual Learning," a lecture note providing an in-depth explanation of the OGD method.

  * **`notebooks/`**: Includes Jupyter notebooks with practical implementations and demonstrations of OGD.

      * [`model-0.ipynb`](https://www.google.com/search?q=uploaded:model-0.ipynb): Demonstrates catastrophic forgetting in a simple sequential learning task (MNIST digits 0-4 then 5-9) without OGD.
      * [`model-1.ipynb`](https://www.google.com/search?q=uploaded:model-1.ipynb): Compares the performance of Standard SGD and OGD on sequential regression tasks, showcasing OGD's ability to preserve knowledge.
      * [`model-2.ipynb`](https://www.google.com/search?q=uploaded:model-2.ipynb): Further experiments with Standard SGD and OGD on different regression tasks.

## Key Concepts

### Catastrophic Forgetting

Deep neural networks often suffer from catastrophic forgetting when trained sequentially on new tasks. This phenomenon leads to a drastic decrease in performance on previously learned tasks after the model is updated with new data. For instance, a model trained to classify digits 0-4 will lose its ability to do so after being trained on digits 5-9 using a simple training approach.

### Orthogonal Gradient Descent (OGD)

OGD is a method introduced to combat catastrophic forgetting. It works by projecting the gradients from new tasks onto a subspace that ensures the neural network's output on previous tasks remains unchanged. This projection allows the model to learn new tasks effectively while retaining knowledge from prior tasks, without the need to store old data, which can raise privacy concerns.

## Experiments and Results

The efficacy of OGD is demonstrated through various experiments, primarily focusing on sequential learning scenarios.

### MNIST Digit Classification

In an experiment detailed in `mini-report.pdf` and `model-0.ipynb`, a simple neural network was trained on the MNIST dataset divided into two sequential tasks:

  * **Task A**: Classifying digits 0-4.
  * **Task B**: Classifying digits 5-9.

**Results with Simple Training (without OGD):**

  * After training on Task A, the model achieved a test accuracy of 97.55% on Task A.
  * However, after training on Task B, the test accuracy on Task A dropped significantly to 0%. This clearly illustrates catastrophic forgetting.

**Results with OGD:**
While `model-0.ipynb` specifically shows the catastrophic forgetting without OGD, the `report.pdf` discusses OGD's effectiveness in preserving knowledge. The `model-1.ipynb` further demonstrates OGD's advantage in a regression context.

### Regression Tasks with Synthetic Data

`model-1.ipynb` and `model-2.ipynb` present experiments comparing Standard SGD and OGD on synthetic regression tasks (e.g., approximating `x`, `1.05 * x + 0.05`, and `x^2`). The Mean Squared Error (MSE) on Task 1 is monitored after training on subsequent tasks.

**Comparative Results:**

  * **Standard SGD:** When training on Task 2 (similar) after Task 1, the MSE on Task 1 increased from 0.000183 to 0.362654. After training on Task 3 (different), the MSE on Task 1 further rose to 1.026326. This indicates a significant loss of performance on Task 1.
  * **OGD:** With OGD, the MSE on Task 1 after training on Task 2 was 0.099431, and after training on Task 3, it was 0.146633. This demonstrates that OGD considerably limits the increase in MSE on previous tasks, effectively preserving learned knowledge.

## Usage

The experiments can be reproduced by running the Jupyter notebooks located in the `notebooks/` directory.

### Dependencies

  * PyTorch
  * Torchvision
  * Matplotlib

### How to Run

1.  Ensure you have the necessary dependencies installed.
2.  Open the `.ipynb` files using Jupyter Notebook or JupyterLab.
3.  Run the cells sequentially to execute the experiments and observe the results.

## Author

  * Ali Ahmadi Esfidi

## Date

  * May 28, 2025 (8 Khordad 1404)
