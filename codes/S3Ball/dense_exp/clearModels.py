import os

GROUPS = [
    'symm_0-vae', 
    'symm_4-ae' , 
    'symm_4-vae', 
]
RAND_INIT_IDS = [16, 42, 100]

def main():
    print('This will clear all training progress in ./dense_exp')
    print('Continue? y/n')
    if input().lower() != 'y':
        print('Cancelled.')
        return
    assert set(os.listdir()) == set(GROUPS + ['clearModels.py'])
    for group in GROUPS:
        os.chdir(group)
        for rand_init_id in RAND_INIT_IDS:
            os.chdir(str(rand_init_id))
            print('removing', group, rand_init_id)
            os.system('rm -r ./*')
            os.chdir('..')
        os.chdir('..')
    print('ok')

main()
