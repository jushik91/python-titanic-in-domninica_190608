from flask import Flask
from flask import render_template, request, jsonify
import re
from calculator.controller import CalculatorController


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/move/<path>")
def move_ui_calc(path):
    return render_template('{}.html'.format(path))


@app.route("/ui_calc")
def ui_calc():
    stmt = request.args.get('stmt','NONE')
    if(stmt == 'NONE'):
        print('넘어온 값이 없음')
    else:
        print('넘어온식 {}'.format(stmt))
        patt = '[0-9]+'
        op = re.sub(patt,'', stmt)
        print('넘어온연산자 {}'.format(op))
        nums = stmt.split(op)
        result = 0
        n1 = int(nums[0])
        n2 = int(nums[1])
        if op == '+':
            result = n1 + n2
        elif op == '-':
            result = n1 - n2
        elif op == '*':
            result = n1 * n2
        elif op == '/':
            result = n1 / n2

    return jsonify(result = result)

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def sum(self):
        return self.first+self.second

    def sub(self):
        return self.first - self.second

    def mul(self):
        return self.first * self.second

    def div(self):
        return self.first / self.second


@app.route('/move/<path>')
def move_ai_calc(path):
    return render_template('{}.html'.format(path))

@app.route('/ai_calc', methods=["POST"])
def ai_calc():
    num1 = request.form['num1']
    num2 = request.form['num2']
    opcode = request.form['opcode']
    c = CalculatorController(num1, num2, opcode)
    result = c.calc()
    render_params = {}
    render_params['result'] = result
    return render_template('ai_calc.html', **render_params)

@app.route('/move/<path>')
def move_titanic(path):
    return render_template('{}.html'.format(path))



if __name__ == '__main__':
    app.run()
