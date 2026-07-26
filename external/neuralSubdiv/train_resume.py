from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *
from models import *

import random
import math

NETPARAMS = 'netparams.dat'
CHECKPOINT = 'checkpoint_latest.pt'


def torch_load(path, device):
    try:
        return torch.load(path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device(device))


def atomic_torch_save(data, path):
    tmp_path = path + '.tmp'
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, net, optimizer, next_epoch, best_loss, train_loss_his,
                    valid_loss_his, completed=False, stopped_early=False):
    atomic_torch_save({
        'next_epoch': next_epoch,
        'best_loss': best_loss,
        'train_loss_his': train_loss_his,
        'valid_loss_his': valid_loss_his,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': get_rng_state(),
        'completed': completed,
        'stopped_early': stopped_early,
    }, path)


def write_loss_history(params, train_loss_his, valid_loss_his):
    np.savetxt(params['output_path'] + 'train_loss.txt',
               np.array(train_loss_his), delimiter=',')
    np.savetxt(params['output_path'] + 'valid_loss.txt',
               np.array(valid_loss_his), delimiter=',')


def write_validation_outputs(params, T, net):
    mIdx = 0
    x = T.getInputData(mIdx)
    outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    tgp.writeOBJ(params['output_path'] + str(mIdx) + '_oracle.obj',
                 T.meshes[mIdx][len(outputs) - 1].V.to('cpu'),
                 T.meshes[mIdx][len(outputs) - 1].F.to('cpu'))
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) + '_subd' + str(ii) + '.obj',
                     x, T.meshes[mIdx][ii].F.to('cpu'))

    mIdx = 0
    x = T.getInputData(mIdx)
    outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    dV = torch.rand(1, 3).to(params['device'])
    R = random3DRotation().to(params['device'])
    x[:, :3] = x[:, :3].mm(R.t())
    x[:, 3:] = x[:, 3:].mm(R.t())
    x[:, :3] += dV
    outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) + '_rot_subd' + str(ii) + '.obj',
                     x, T.meshes[mIdx][ii].F.to('cpu'))


def load_best_model_if_available(params, net):
    best_path = params['output_path'] + NETPARAMS
    if os.path.exists(best_path):
        net.load_state_dict(torch_load(best_path, params['device']))
        print('loaded best model for final outputs: %s' % best_path, flush=True)
        return True
    print('warning: best model not found; using current network for final outputs',
          flush=True)
    return False


def main():
    folder = sys.argv[1]
    if not folder.endswith('/'):
        folder += '/'

    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)
    os.makedirs(params['output_path'], exist_ok=True)

    if params['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('hyperparameters.json requests cuda, but CUDA is not available')

    S = pickle.load(open(params['train_pkl'], 'rb'))
    S.computeParameters()
    S.toDevice(params['device'])

    T = pickle.load(open(params['valid_pkl'], 'rb'))
    T.computeParameters()
    T.toDevice(params['device'])

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    net = SubdNet(params)
    net = net.to(params['device'])
    net.apply(init_weights)

    lossFunc = torch.nn.MSELoss().to(params['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    checkpoint_path = params['output_path'] + CHECKPOINT
    trainLossHis = []
    validLossHis = []
    bestLoss = np.inf
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch_load(checkpoint_path, params['device'])
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_to(optimizer, params['device'])
        set_rng_state(checkpoint.get('rng_state'))
        start_epoch = int(checkpoint['next_epoch'])
        bestLoss = checkpoint['best_loss']
        trainLossHis = list(checkpoint.get('train_loss_his', []))
        validLossHis = list(checkpoint.get('valid_loss_his', []))
        print('resuming from checkpoint: epoch %d / %d, best valid %.6e'
              % (start_epoch, params['epochs'], bestLoss), flush=True)
    else:
        print('starting fresh training run', flush=True)

    try:
        for epoch in range(start_epoch, params['epochs']):
            ts = time.time()

            trainErr = 0.0
            for mIdx in range(S.nM):
                x = S.getInputData(mIdx)
                outputs = net(x, mIdx, S.hfList, S.poolMats, S.dofs)

                Vt = S.meshes[mIdx][params['numSubd']].V.to(params['device'])

                loss = 0.0
                for ii in range(params['numSubd'] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        'non-finite train loss at epoch %d, mesh %d' %
                        (epoch, mIdx))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        'non-finite train loss value at epoch %d, mesh %d' %
                        (epoch, mIdx))
                trainErr += loss_value
            trainLossHis.append(trainErr / S.nM)

            validErr = 0.0
            for mIdx in range(T.nM):
                x = T.getInputData(mIdx)
                outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

                Vt = T.meshes[mIdx][params['numSubd']].V.to(params['device'])

                loss = 0.0
                for ii in range(params['numSubd'] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])

                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        'non-finite valid loss at epoch %d, mesh %d' %
                        (epoch, mIdx))
                validErr += loss_value
            validLossHis.append(validErr / T.nM)

            if validErr < bestLoss:
                bestLoss = validErr
                torch.save(net.state_dict(), params['output_path'] + NETPARAMS)

            write_loss_history(params, trainLossHis, validLossHis)
            save_checkpoint(checkpoint_path, net, optimizer, epoch + 1, bestLoss,
                            trainLossHis, validLossHis)

            print('epoch %d, train loss %.6e, valid loss %.6e, remain time: %s'
                  % (epoch, trainLossHis[-1], validLossHis[-1],
                     int(round((params['epochs'] - epoch) * (time.time() - ts)))),
                  flush=True)
            print('checkpoint saved: %s' % checkpoint_path, flush=True)

    except KeyboardInterrupt:
        print('interrupted; latest completed epoch checkpoint is already saved at %s'
              % checkpoint_path, flush=True)
        raise
    except FloatingPointError as exc:
        print('stopping early: %s' % exc, flush=True)
        write_loss_history(params, trainLossHis, validLossHis)
        if not os.path.exists(params['output_path'] + NETPARAMS):
            raise
        load_best_model_if_available(params, net)
        save_checkpoint(checkpoint_path, net, optimizer, params['epochs'],
                        bestLoss, trainLossHis, validLossHis,
                        completed=True, stopped_early=True)
        write_validation_outputs(params, T, net)
        print('early stop complete; final outputs use best saved model',
              flush=True)
        return

    write_loss_history(params, trainLossHis, validLossHis)
    load_best_model_if_available(params, net)
    save_checkpoint(checkpoint_path, net, optimizer, params['epochs'], bestLoss,
                    trainLossHis, validLossHis, completed=True)
    write_validation_outputs(params, T, net)
    print('training complete; final checkpoint saved: %s' % checkpoint_path, flush=True)


if __name__ == '__main__':
    main()
