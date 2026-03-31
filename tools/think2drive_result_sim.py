import numpy as np

driving_scores = []
num_infractions = []

for i in range(10000):
  infractions = 0
  for i in range(90):
    sample = np.random.uniform()
    if sample > 0.84:
      infractions += 1

  infraction_score = 0.65**infractions

  driving_scores.append(100 * infraction_score)
  num_infractions.append(infractions)

print('Avg. Driving Score: ', np.mean(driving_scores))
print('Avg. Num Infractions: ', np.mean(num_infractions))
