eps = 1e-4
n_generations = N_generations_max
for i in range(N_generations_max-1):
#     print(i)
    SR_i = SR_all[i]
    metric_i = metric_all[i]
    if metric_i.min() ==  metric_i.max():
        n_generations = i+1
        break
    selected_inds = selection_tournament(-metric_i, N_population, 2, elitism=True)
    SR_i = [SR_i[ind] for ind in selected_inds] 
    SR_ip = crossover_pop(SR_i, -metric_i[selected_inds], crossover_operator, k=5)
    metric_ip = np.array([metric(SR_i_j , interpolate_SR, [depth, y], muy_k+p0, metric_type=metric_type) for SR_i_j in SR_ip])
    SR_ip = mutation_v2_pop(SR_ip, -metric_ip, xrange=[0,1], yrange=[0,3], eta=20)
    metric_ip = np.array([metric(SR_i_j , interpolate_SR, [depth, y], muy_k+p0, metric_type=metric_type) for SR_i_j in SR_ip])
    metric_all[i+1] = metric_ip
    SR_all += [SR_ip]
print(n_generations)
# metric_all = metric_all[:n_generations]
SR_all = SR_all[:n_generations]
