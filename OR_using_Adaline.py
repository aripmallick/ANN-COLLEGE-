# Bipolar OR gate training data: (x1, x2, target)
data = [
    (1, 1, 1),
    (1, -1, 1),
    (-1, 1, 1),
    (-1, -1, -1)
]

# Step 1: Initialize weights, bias, learning rate, and error limit
w1 = 0.1
w2 = 0.1
b = 0.1
alpha = 0.1
error_limit = 2.0

epoch = 0

# Step 2: Train until total squared error <= error_limit
while True:
    epoch = epoch+1
    total_error = 0
    print(f"\n--- Epoch {epoch} ---")

    # Step 3 to 6: Loop through each row
    for x1, x2, t in data:
        # Step 5: Compute net input (linear activation: f(y_in) = y_in)
        y_in = (w1 * x1) + (w2 * x2) + b

        # Step 6: Update weights, bias, and squared error
        diff = t - y_in
        w1 = w1 + alpha * diff * x1
        w2 = w2 + alpha * diff * x2
        b = b + alpha * diff

        error = diff ** 2 
        total_error = total_error+error

        print(f"Input: ({x1}, {x2}) | y_in: {y_in:.4f} | w1: {w1:.4f}, w2: {w2:.4f}, b: {b:.4f} | Error: {error:.2f}")

    print(f"Total Squared Error = {total_error:.4f}")

    # Step 7: Check stopping condition
    if total_error <= error_limit:
        print(f"\nTraining stopped at Epoch {epoch}")
        break

# Testing the trained ADALINE model using bipolar threshold
print(f"Final Weights: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}\n")

for x1, x2, t in data:
    y_in = (w1 * x1) + (w2 * x2) + b
    if y_in >=0:
        out=1
    else:
        out=-1
    
    print(f"Input: ({x1}, {x2}) -> y_in: {y_in:.4f} -> Output Y: {out} (Target: {t})")