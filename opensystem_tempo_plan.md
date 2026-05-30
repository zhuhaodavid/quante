有一个问题 HamiltonianDynamics 的 real-time 下scale 对应的是 -1j， 但 LindbladDynamics 理论上也是 real-time，但对应的 scale 如果有的话应该是 1
这虽然没有问题大， 有点奇怪
HamiltonianDynamics 的 traceA 也应该在运行的过程中计算
EigenBackend这里首先 matrix = generator.generator(kind=kind, sparse=False) 有问题，应该和ExpmMultiplyBackend类似的处理，所以确实 _operator_data 可以单独拿出来作为一个函数
然后根据这个结果来判断eigen采用的方案以及_evolve_from_eigensystem_*和 _reconstruct_from_eigenbasis的scale
all_states有点不和谐，是不是可以删了，然把EigenBackend的measure写到E:\hzhu\library\quante\quante\linalg\evolve\eigen_evolve.py里面
这样会失去 EigenBackend 的优势，但为了 EvolveEngine 的一致性，这应该是值得的，EigenBackend 本来的优势也应该放在 eigen_evolve.py 里面，而 EvolveEngine 需要保持格式的一致性
以及把 EvolveEngine 的measure中 bulk_measure 也可以移到 eigen_evolve 里面
EigenBackend 的__init__会不会有点长？
pre_obs 这个名字有点丑