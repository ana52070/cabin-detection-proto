# -*- coding: utf-8 -*-
import sys

def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else 'false'
    enable_val = 'True' if target in ('true', '1', 'on') else 'False'
    
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if enable_val == 'False':
        content = content.replace('"enabled": True,', '"enabled": False,')
    else:
        content = content.replace('"enabled": False,', '"enabled": True,')

    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Set config.py KALMAN_CONFIG['enabled'] = {enable_val}")

if __name__ == '__main__':
    main()
