from scipy.stats import ttest_ind_from_stats

mean1 = 2.7
std_dev1 = 1.1
n1 = 6

mean2 = 2.88
std_dev2 = .12
n2 = 10

# Esegui il t-test
t_stat, p_value = ttest_ind_from_stats(mean1, std_dev1, n1, mean2, std_dev2, n2)
print(t_stat, p_value)