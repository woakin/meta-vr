import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { BUDGET_STEPS, OVERLOAD_EMS_MS, RECOVER_EMS_MS, nextBudgetIndex } from '../src/house-of-cards.ts';

describe('nextBudgetIndex', () => {
  test('steps down one level when overloaded', () => {
    assert.equal(nextBudgetIndex(OVERLOAD_EMS_MS + 1, 0), 1);
    assert.equal(nextBudgetIndex(OVERLOAD_EMS_MS + 5, 1), 2);
  });
  test('clamps at the lowest budget when overloaded', () => {
    assert.equal(nextBudgetIndex(1000, BUDGET_STEPS.length - 1), BUDGET_STEPS.length - 1);
  });
  test('steps up one level with headroom', () => {
    assert.equal(nextBudgetIndex(RECOVER_EMS_MS - 1, 2), 1);
    assert.equal(nextBudgetIndex(5, 1), 0);
  });
  test('clamps at the highest budget with headroom', () => {
    assert.equal(nextBudgetIndex(0, 0), 0);
  });
  test('holds between the hysteresis lines', () => {
    assert.equal(nextBudgetIndex(10, 2), 2);
    assert.equal(nextBudgetIndex(10, 0), 0);
    assert.equal(nextBudgetIndex(10, BUDGET_STEPS.length - 1), BUDGET_STEPS.length - 1);
  });
});
