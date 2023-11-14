conditions = {0: observation}
    samples = None
    obs = [observation,observation,observation,observation,observation]
    observations = np.zeros((5, 1024, 66))
    actions = np.zeros((5, 1024, 0))
    for n in range(SAMPLING_NUM):
        observations[n,0] = observation
    for i in range(1,1024):
        
        action, temp_samples = policy(conditions, batch_size=SAMPLING_NUM, verbose=args.verbose)
        for n in range(SAMPLING_NUM):
            obs[n] = update_heuristics(obs[n], temp_samples.observations[n,1])
            observations[n,i] = obs[n]

            # for n in range(SAMPLING_NUM):
            # if samples is None:
            # # first iteration, assign the selected temp_samples to the samples object
            #     samples = Trajectories(
            #         actions=(temp_samples.actions[0,1]),
            #         observations = np.array([obs),
            #         values= torch.tensor(temp_samples.values[0,1])
            #     )
            # else:
            #     # Concatenate the new ith temp_samples with the existing samples
            #     samples = Trajectories(
            #         actions=np.concatenate((samples.actions, (temp_samples.actions[0,1])), axis=0),
            #         observations=np.concatenate((samples.observations, (obs)), axis=0),
            #         values= samples.values
            #     )
        #replace starting condition (idea 1)
            conditions = {0: obs[n]}
        #add in condition (idea 2)
        # conditions[i] = obs
    samples = Trajectories(
                    actions=actions,
                    observations = observations,
                    values= torch.empty((5,))
                )

            if samples is None:
            # first iteration, assign the selected temp_samples to the samples object
                samples = Trajectories(
                    actions=(temp_samples.actions[0,1]),
                    observations = obs,
                    values= torch.tensor(temp_samples.values[0])
                )
            else:
                # Concatenate the new ith temp_samples with the existing samples
                samples = Trajectories(
                    actions=np.concatenate((samples.actions, (temp_samples.actions[0,1])), axis=0),
                    observations=np.concatenate((samples.observations, (obs)), axis=0),
                    values= samples.values
                )