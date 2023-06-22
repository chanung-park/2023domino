from pennylane import numpy as np
import pennylane as qml 



# Hamiltonian definition H = sum_i (- Z_i * Z_{i+1})
def generate_Hamiltonian(N):
    coeffs = -1 * np.ones((N,))
    obs = [qml.PauliZ(i)@qml.PauliZ(i+1) for i in range(N-1)]
    obs.append(qml.PauliZ(N-1)@qml.PauliZ(0))
    H = qml.Hamiltonian(coeffs, obs)
    print(H)
    return H 

def find_groundstate(N, p, state_circuit, H, max_iterations = 300, conv_tol = 1e-06):
    # call device 
    dev = qml.device("default.qubit", wires = N )
    
    #define cost function find expectation value of Hamiltonian which is energy for parametrized state 
    @qml.qnode(dev, interface='autograd')
    def cost_fn(param):
        state_circuit(param, w = range(N))
        return qml.expval(H)
    # define optimizer
    opt = qml.GradientDescentOptimizer(stepsize = 0.1 )
    # start with random parameter 
    param = np.random.randn(p, 2*N, requires_grad = True) # size is define through state circuit 

    energy = [cost_fn(param)]
    for n in range(max_iterations):
        param, prev_energy = opt.step_and_cost(cost_fn, param)
        energy.append(cost_fn(param))
        conv = np.abs(energy[-1]-prev_energy)
        if conv <= conv_tol:
            break
    return energy, param

if __name__=='__main__':
    N = 5
    p = 3 
    # variational circuit for state 

    def circuit(param, # p X (N+N) thetas and phis  
                w #wires list 
                ):
        (p, _)  = param.shape
        N = len(w)
        # prepare |+> state with Hadamard gate 
        for i in range(N):
            qml.Hadamard(wires = w[i])
        # parametrized circuit with depth p 
        for step in range(p):
            for i in range(N):
                qml.RX(param[step, i], wires = w[i]) # exp(-i X phi/2)
            for i in range(N-1):
                qml.CNOT(wires = w[i: i+2])
                qml.RZ(param[step, i], wires = w[i+1])
                qml.CNOT(wires = w[i: i+2])
            qml.CNOT(wires = [w[N-1], w[0]])
            qml.RZ(param[step, N-1], wires = w[N-1])
            qml.CNOT(wires = [w[N-1], w[0]])

    H = generate_Hamiltonian(N)
    energy_list, param = find_groundstate(N, p, circuit, H)
